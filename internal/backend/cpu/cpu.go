// Package cpu 实现了纯 Go 的 CPUBackend。
//
// 这是 Minfer 的第一个（也是最基本的）计算后端。
// 所有运算都用最朴素的 for 循环实现——没有任何 SIMD 优化、
// 没有 BLAS 调用、没有 GPU。这样做的目的是：
//
//   1. 每一行数学都看得懂——三重循环的矩阵乘法、逐元素 Softmax
//   2. 作为后续优化的基线——优化前和优化后结果对比
//   3. 学习价值——理解了朴素实现，才能理解为什么需要优化
//
// 性能预期（纯 Go，单核，float32）：
//   - 0.5B 模型 (Qwen2.5): 30-50 tok/s
//   - 3B 模型: 8-12 tok/s
//   - 7B 模型: 3-5 tok/s
//
// 后续优化方向：
//   Phase 2: goroutine 并行 attention heads
//   Phase 3: cache-aware tiled matmul
//   Phase 4: cgo + OpenBLAS 做 MatMul
//   Phase 5: CUDA backend
package cpu

import (
	"fmt"
	"math"

	"github.com/yusiwen/minfer/internal/compute"
	"github.com/yusiwen/minfer/internal/tensor"
)

// 编译期检查：CPUBackend 是否实现了 compute.Backend 接口
var _ compute.Backend = (*CPUBackend)(nil)

// CPUBackend 实现了 compute.Backend 接口。
// 所有方法都用纯 Go 实现。
type CPUBackend struct{}

// New 创建一个新的 CPUBackend 实例。
func New() *CPUBackend {
	return &CPUBackend{}
}

// MatMul 计算 C = A × B，使用最经典的 ijk 三重循环。
//
// 算法原理
// ─────────
//
// 矩阵乘法是 LLM 推理中最核心的运算。一个 7B 模型的推理过程中，
// ~70% 的浮点运算发生在 MatMul 里。
//
// 数学定义：
//   C[i][j] = Σ A[i][k] × B[k][j]   (k = 0..K-1)
//
// 用三重循环实现：
//   for i = 0..M-1:         // 遍历 A 的每一行
//     for j = 0..N-1:       // 遍历 B 的每一列
//       for k = 0..K-1:     // 累乘求和
//         C[i][j] += A[i][k] × B[k][j]
//
// 内存访问模式
// ────────────
// 这个朴素实现的缓存效率不高。原因是 B[k][j] 的访问是列优先的，
// 而 Go（和 C）使用行优先存储，导致每次循环都在不同的 cache line
// 上跳来跳去。
//
// 更好的做法（后续优化）：
//   ikj 顺序: for i → for k → for j
//   这样 A[i][k] 和 B[k][j] 都是顺序访问（一个按行，一个按行片段）
//   缓存友好度大幅提升。
//
// 为什么先写 ijk：
//   最直观，最容易和数学公式对应。
//   先跑通、验证正确，再优化。
func (b *CPUBackend) MatMul(a, bTensor *tensor.Tensor) (*tensor.Tensor, error) {
	// 形状检查
	if a.Dims() != 2 || bTensor.Dims() != 2 {
		panic("cpu.MatMul: inputs must be 2D tensors")
	}
	M := a.Shape[0]
	K := a.Shape[1]
	if bTensor.Shape[0] != K {
		panic(fmt.Sprintf(
			"cpu.MatMul: shape mismatch: A[%d,%d] cannot multiply B[%d,%d] (inner dims %d != %d)",
			M, K, bTensor.Shape[0], bTensor.Shape[1], K, bTensor.Shape[0],
		))
	}
	N := bTensor.Shape[1]

	// 创建输出张量 C: [M, N]（初始值为 0）
	c := tensor.New(M, N)

	// 三重循环：最朴素的 ijk
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				// 计算偏移量（行优先）
				// a[i][k] 的偏移量 = i*K + k
				// b[k][j] 的偏移量 = k*N + j
				sum += a.Data[i*K+k] * bTensor.Data[k*N+j]
			}
			c.Data[i*N+j] = sum
		}
	}

	return c, nil
}

// Softmax 在最后一个维度上计算 Softmax（原地修改）。
//
// 算法原理
// ────────
//
// Softmax 将一组任意实数转换成一个概率分布（所有值 ∈ [0,1]，和为 1）。
//
// 朴素版本（数值不稳定——可能在 exp 时溢出）：
//   Softmax(x_i) = exp(x_i) / Σ(exp(x_j))
//
// 数值稳定版本（先减去最大值）：
//   1. 找到当前行最大值 max_val
//   2. 每个元素: x_i ← exp(x_i - max_val)
//   3. 求和: sum = Σ(x_i)
//   4. 每个元素: x_i ← x_i / sum
//
// 为什么要减 max：
//   float32 的最大值约 3.4×10³⁸，exp(88) 就接近这个值了。
//   如果 x 中有一个很大的值（比如 100），exp(100) 立即溢出。
//   减去最大值后，最大的 exp 项变成 exp(0) = 1，所有项都在安全范围内。
//
// 在注意力计算中的角色：
//   scores = Softmax(Q × K^T / √d_k)
//   这步之后，scores 的每一行变成一个概率分布，
//   告诉模型"当前位置应该关注序列中的哪些位置"。
//
// 注意：该实现永远返回 nil，保留 error 返回值是为了与 compute.Backend
// 接口一致。未来 GPU 后端可能因为设备错误返回非 nil error。
func (b *CPUBackend) Softmax(t *tensor.Tensor) error {
	// 获取最后一个维度的大小
	// 对于形状 [M, N] 的矩阵，lastDim = N
	// 对于形状 [B, H, S, S] 的注意力分数，lastDim = S
	n := t.Shape[t.Dims()-1]

	// 外层遍历所有"行"
	// 行数 = 总元素数 / lastDim
	rows := t.NumElements() / n

	for r := 0; r < rows; r++ {
		// 计算行的起始和结束偏移量
		start := r * n
		end := start + n

		// Step 1: 找到当前行的最大值（用于数值稳定）
		maxVal := t.Data[start]
		for i := start + 1; i < end; i++ {
			if t.Data[i] > maxVal {
				maxVal = t.Data[i]
			}
		}

		// Step 2: 计算 exp(x_i - max) 并求和
		var sum float32
		for i := start; i < end; i++ {
			t.Data[i] = float32(math.Exp(float64(t.Data[i] - maxVal)))
			sum += t.Data[i]
		}

		// Step 3: 除以和（归一化到概率分布）
		for i := start; i < end; i++ {
			t.Data[i] /= sum
		}
	}

	return nil
}

// RMSNorm 执行 RMS 归一化。
//
// 算法原理
// ────────
//
// RMSNorm 是 Layer Normalization 的简化版本。
//
// LayerNorm 公式：
//   y = (x - μ) / σ × γ + β
//   其中 μ = mean(x), σ = sqrt(var(x) + ε)
//
// RMSNorm 公式（去掉了均值 μ 的计算）：
//   y = x / RMS(x) × γ
//   其中 RMS(x) = sqrt( (1/N) × Σ(x_j²) + ε )
//
// RMSNorm 省略了均值计算，因此比 LayerNorm 快约 30%，
// 且实验表明在 Transformer 中效果与 LayerNorm 相当。
//
// 参数 weight（γ）是可学习的缩放向量。
// 每个特征维度有一个缩放因子。
func (b *CPUBackend) RMSNorm(t, weight *tensor.Tensor) error {
	// 最后一个维度的大小 = hidden_dim
	n := t.Shape[t.Dims()-1]
	rows := t.NumElements() / n

	// weight 必须是一维向量，长度等于 hidden_dim
	if weight.Dims() != 1 || weight.NumElements() != n {
		panic(fmt.Sprintf(
			"cpu.RMSNorm: weight shape mismatch: got (%d,) but input has hidden_dim %d",
			weight.NumElements(), n,
		))
	}

	// TODO: 从 model.Config 中读取 epsilon，当前硬编码
	const epsilon = 1e-6 // 防除零

	for r := 0; r < rows; r++ {
		start := r * n
		end := start + n

		// Step 1: 计算 RMS
		// RMS = sqrt( (1/N) × Σ(x_i²) + ε )
		var sumSq float32
		for i := start; i < end; i++ {
			sumSq += t.Data[i] * t.Data[i]
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(n) + epsilon)))

		// Step 2: 归一化并乘以可学习权重
		// y_i = x_i / RMS(x) × weight_i
		for i := start; i < end; i++ {
			t.Data[i] = (t.Data[i] / rms) * weight.Data[i-start]
		}
	}

	return nil
}

// RoPE 对 Q 和 K 施加旋转位置编码。
//
// 算法原理
// ────────
//
// 为什么需要位置编码？
//   Self-attention 本身是位置无关的——交换两个 token 的位置，
//   注意力分数不会变。所以需要额外的位置信息。
//
// RoPE 的核心思想：
//   把 (q_2i, q_{2i+1}) 看作二维平面上的一个向量，
//   根据位置 pos 旋转这个向量。旋转矩阵为：
//
//   [cos(pos·θ_i)  -sin(pos·θ_i)]
//   [sin(pos·θ_i)   cos(pos·θ_i)]
//
//   其中 θ_i = base^(-2i/d)，base = 10000
//   （这里跟 Transformer 原版的位置编码用的 base 是一样的）
//
// 旋转后：
//   q'_2i   = q_2i × cos(θ) - q_{2i+1} × sin(θ)
//   q'_{2i+1} = q_{2i+1} × cos(θ) + q_2i × sin(θ)
//
// 频率递减设计
// ──────────
//   i = 0（最早的对）: θ_0 = 10000^0 = 1 → 旋转最快（编码短距离）
//   i = d/2-1（最后一对）: θ ≈ 10000^(-1) ≈ 1/10000 → 旋转最慢（编码长距离）
//
// 这种频率递减的设计和 Transformer 原版的 Sinusoidal PE 完全一致。
// 靠前的维度编码短距离信息，靠后的维度编码长距离信息。
//
// 在 minfer 中，每次只处理一个 token（decode 阶段），
// 所以 q 和 k 的 seq_len 维度为 1。
//
// 参数：
//   q:   Query 当前 token，形状 [1, num_heads, head_dim]
//   k:   Key 当前 token，形状 [1, num_heads, head_dim]
//   pos: 当前 token 在序列中的位置
//   dim: head_dim（如 64 或 128）
func (b *CPUBackend) RoPE(q, k *tensor.Tensor, pos, dim int) error {

	// 形状检查：q 和 k 必须是 3 维 [1, num_heads, dim]
	if q.Dims() != 3 || k.Dims() != 3 {
		panic("cpu.RoPE: q and k must be 3D tensors [1, num_heads, head_dim]")
	}
	if q.Shape[2] != dim || k.Shape[2] != dim {
		panic(fmt.Sprintf(
			"cpu.RoPE: head_dim mismatch: q has %d, k has %d, expected %d",
			q.Shape[2], k.Shape[2], dim,
		))
	}
	if q.Shape[1] != k.Shape[1] {
		panic(fmt.Sprintf(
			"cpu.RoPE: num_heads mismatch: q has %d, k has %d",
			q.Shape[1], k.Shape[1],
		))
	}
	if dim%2 != 0 {
		panic(fmt.Sprintf(
			"cpu.RoPE: head_dim must be even, got %d",
			dim,
		))
	}

	// TODO: 从 model.Config 中读取 base，当前硬编码
	const base = 10000.0

	numHeads := q.Shape[1]
	headDim := dim

	for h := 0; h < numHeads; h++ {
		for i := 0; i < headDim; i += 2 {
			// 当前维度对在两个 head 中的偏移量
			offset := h*headDim + i

			// 计算 θ_i
			// 公式: θ_i = base^(-2i/d) = 1 / base^(2i/d)
			freq := 1.0 / math.Pow(base, float64(i)/float64(headDim))

			// 计算 cos 和 sin
			cosVal := float32(math.Cos(float64(pos) * freq))
			sinVal := float32(math.Sin(float64(pos) * freq))

			// 对 Q 施加旋转
			q0 := q.Data[offset]
			q1 := q.Data[offset+1]
			q.Data[offset] = q0*cosVal - q1*sinVal
			q.Data[offset+1] = q1*cosVal + q0*sinVal

			// 对 K 施加旋转（同样的 cos/sin）
			k0 := k.Data[offset]
			k1 := k.Data[offset+1]
			k.Data[offset] = k0*cosVal - k1*sinVal
			k.Data[offset+1] = k1*cosVal + k0*sinVal
		}
	}

	return nil
}

// Silu 计算 SiLU 激活函数（原地修改）。
//
// 数学定义：
//   SiLU(x) = x × sigmoid(x) = x / (1 + exp(-x))
//
// SiLU 也叫 Swish（Google 2017 年提出），
// 是 ReLU 的平滑版本：
//
//   ReLU:    x > 0 → x, x ≤ 0 → 0
//   SiLU:    负半轴有微小负值，0 附近平滑过渡
//
// 平滑梯度的好处：
//   ReLU 在 x=0 处不可导，可能导致神经元"死亡"
//   （一旦进入负半轴就永远输出 0，梯度为 0，彻底不学习）。
//   SiLU 在 0 附近有连续导数，不会完全死亡。
//
// 在 SwiGLU FFN 中的角色：
//   gate = SiLU(x × W_gate)  ← 这是"门控"信号
//   up   = x × W_up           ← 这是"内容"信号
//   out  = (gate × up) × W_down  ← 门控×内容后投影回 hidden_dim
//
// 门控机制让 FFN 可以"决定"哪些信息通过、哪些被抑制。
// 这是 SwiGLU 比标准 ReLU FFN 效果更好的原因。
//
// 注意：该实现永远返回 nil，保留 error 返回值是为了与 compute.Backend
// 接口一致。未来 GPU 后端可能因为设备错误返回非 nil error。
func (b *CPUBackend) Silu(t *tensor.Tensor) error {
	for i := range t.Data {
		// sigmoid(x) = 1 / (1 + exp(-x))
		t.Data[i] = t.Data[i] / (1 + float32(math.Exp(-float64(t.Data[i]))))
	}
	return nil
}

// Add 执行逐元素加法：c = a + b
//
// a 和 b 必须有相同形状。
// Panic 条件：a 和 b 形状不同。
//
// 在 Transformer 中用于残差连接（Residual Connection）：
//   output = sub_layer(output) + output
//
// 残差连接的意义：
//   如果没有残差连接，深层网络容易出现"梯度消失"问题——
//   梯度经过多层反向传播后趋近于 0，浅层权重几乎得不到更新。
//   残差连接让梯度有一条"高速公路"直达浅层。
func (b *CPUBackend) Add(a, bTensor *tensor.Tensor) (*tensor.Tensor, error) {
	if len(a.Shape) != len(bTensor.Shape) {
		panic("cpu.Add: tensors have different number of dimensions")
	}
	for i := range a.Shape {
		if a.Shape[i] != bTensor.Shape[i] {
			panic("cpu.Add: tensors have different shapes")
		}
	}
	c := tensor.New(a.Shape...)
	for i := range a.Data {
		c.Data[i] = a.Data[i] + bTensor.Data[i]
	}
	return c, nil
}
