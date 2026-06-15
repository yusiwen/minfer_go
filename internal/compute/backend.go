// Package compute 定义了 Minfer 的计算后端接口。
//
// 这是整个推理引擎的架构核心。所有数值计算都通过这个接口完成，
// 而不是直接在 Tensor 上调用方法。这样设计的目的：
//
//   1. 模型代码（internal/model/）和推理循环（internal/infer/）
//      完全不知道计算在哪运行——CPU、GPU 还是其他硬件。
//
//   2. 新增一个后端只需要实现 Backend 接口，
//      不需要改动任何模型或推理代码。
//
//   3. 学习目的：从纯 Go 的 CPUBackend 开始，每一行代码都自己能看懂。
//      后续可以逐步添加 SIMD 优化、CUDA 后端等。
//
// 后端选择在运行时决定（通过命令行 --backend 参数）：
//
//     minfer --backend cpu run model.gguf   → CPUBackend（默认）
//     minfer --backend cuda run model.gguf  → 未来：CUDABackend
//     minfer --backend metal run model.gguf → 未来：MetalBackend
//
// 题外话——"Kernel" 是什么？
// 在高性能计算（HPC）领域，"kernel" 指的是一段被反复调用的、最核心的计算循环。
// 比如矩阵乘法的三重循环最内层、Softmax 的 exp/sum/div 循环。
// GPU 编程借用了这个术语：一个 GPU kernel 就是在 GPU 上并行执行的这样一段函数。
// 详见 Wolai 页面 "Kernel — From OS to GPU Programming"。
package compute

import "github.com/yusiwen/minfer/internal/tensor"

// Backend 是所有计算后端的统一接口。
//
// 每个方法对应 Transformer 模型中的一个数学运算。
// 所有方法的输入和输出都是 *tensor.Tensor。
type Backend interface {
	// MatMul 计算矩阵乘法: C = A × B
	//
	// 数学定义：
	//   C[i][j] = Σ(A[i][k] × B[k][j]) for k = 0..K-1
	//
	// 形状约束：
	//   A: [M, K], B: [K, N], C: [M, N]
	//
	// 在 Transformer 中，MatMul 是最频繁、最耗时的运算（约占 70% 的推理时间）：
	//   - Q = x × W_q    (hidden_dim × hidden_dim)
	//   - K = x × W_k    (hidden_dim × hidden_dim)
	//   - V = x × W_v    (hidden_dim × hidden_dim)
	//   - scores = Q × K^T  (seq_len × seq_len) — 注意力分数
	//   - output = scores × V  (seq_len × head_dim) — 加权求和
	//   - FFN: gate = x × W_gate, up = x × W_up, down = gate*up × W_down
	//
	// 从零实现的角度，这是第一个值得花时间优化的地方。
	// CPUBackend 永远返回 nil error；GPU 后端可能因设备错误返回非 nil。
	MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error)

	// Softmax 在最后一个维度上计算 Softmax（原地修改）。
	//
	// 数学定义：
	//   Softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x))) for j = 0..N-1
	//
	// 公式中的 "- max(x)" 称为 "数值稳定技巧"：
	// exp 函数在输入很大时会溢出（exp(100) ≈ 2.7×10^43，还在 float32 范围内，
	// 但 exp(710) 就溢出了）。减去最大值后，最大的 exp 项变成 exp(0) = 1，
	// 所有项都在安全范围内。
	//
	// 在注意力计算中：scores = Softmax(Q × K^T / √d_k)
	// 这步之后，scores 的每一行都变成概率分布（和为 1，每个值在 0~1 之间）。
	//
	// 注意：该实现永远返回 nil，保留 error 返回值是为了与 compute.Backend
	// 接口一致。未来 GPU 后端可能因为设备错误返回非 nil error。
	Softmax(t *tensor.Tensor) error

	// RMSNorm 执行 RMS 归一化（Root Mean Square Normalization）。
	//
	// 数学定义：
	//   y_i = x_i × weight_i / RMS(x)
	//   其中 RMS(x) = sqrt( (1/N) × Σ(x_j²) + ε )
	//
	// 与 LayerNorm 的区别：
	//   LayerNorm: 减去均值再除以标准差（需要计算 mean 和 variance）
	//   RMSNorm:   只除以 RMS（省去了均值计算）
	//
	// RMSNorm 是 LayerNorm 的简化版本，计算量少约 30%，
	// 在 LLaMA、Qwen2.5 等现代模型中完全替代了 LayerNorm。
	//
	// 参数：
	//   t:      输入张量 [..., hidden_dim]，会被原地修改
	//   weight: 可学习的缩放参数 [hidden_dim]
	//
	// ε（epsilon）是一个极小的常数（通常 1e-6），防止除以零。
	RMSNorm(t, weight *tensor.Tensor) error

	// RoPE 对 Q 和 K 施加旋转位置编码（Rotary Position Embedding）。
	//
	// 核心思想：通过旋转矩阵对词向量进行变换，使得点积能够编码相对位置信息。
	//
	// 数学公式：
	//   对每个位置 pos 和每对维度 (2i, 2i+1)：
	//     θ_i = 10000^(-2i/d)
	//     q'_2i   = q_2i × cos(pos × θ_i) - q_{2i+1} × sin(pos × θ_i)
	//     q'_{2i+1} = q_{2i+1} × cos(pos × θ_i) + q_2i × sin(pos × θ_i)
	//     K 同理
	//
	// 直观理解：
	//   RoPE 把每一对相邻维度 (2i, 2i+1) 看作二维平面上的一个点，
	//   然后根据位置 pos 旋转这个点。旋转量随维度变化——
	//   靠前的维度旋转慢（编码长距离信息），靠后的维度旋转快（编码短距离信息）。
	//
	// 关键性质（也是 RoPE 被广泛使用的原因）：
	//   位置 pos1 的 Q 和位置 pos2 的 K 做点积时，
	//   结果只依赖于相对位置 (pos1 - pos2)，与绝对位置无关。
	//
	// 参数：
	//   q:   Query 张量 [seq_len, num_heads, head_dim]
	//   k:   Key 张量 [seq_len, num_heads, head_dim]
	//   pos: 当前处理的位置（从 0 开始）
	//   dim: head_dim（从 config 获取）
	//
	// 注意：这里 q 和 k 只需要包含当前这个位置的 slice。
	// 在 decode 阶段，每次只处理一个 token，所以 seq_len=1。
	RoPE(q, k *tensor.Tensor, pos, dim int) error

	// Silu 计算 SiLU 激活函数（Sigmoid Linear Unit，也叫 Swish）。
	//
	// 数学定义：
	//   SiLU(x) = x × σ(x)  其中 σ(x) = 1 / (1 + exp(-x))
	//
	// 图像特征：
	//   负半轴：接近 0（但不是 0，有微小的负值）
	//   正半轴：接近 x（但略小于 x）
	//   在 0 附近平滑过渡（不像 ReLU 在 0 处有尖角）
	//
	// 为什么 SwiGLU 用 SiLU：
	//   SwiGLU 的公式是：FFN_out = (SiLU(x × W_gate) × (x × W_up)) × W_down
	//   "门控"机制：W_gate 的输出经过 SiLU 后，决定 W_up 的输出有多少能通过。
	//   研究表明这种门控机制比标准 ReLU FFN 效果好，但参数量也大了 1/3。
	//
	// 这是原地操作，直接修改输入张量。
	Silu(t *tensor.Tensor) error

	// Add 执行逐元素加法：c = a + b
	//
	// 在 Transformer 中用于残差连接（Residual Connection）：
	//   output = Layer(x) + x
	// 残差连接让梯度能直接流过深层网络，是训练深度模型的关键。
	//
	// CPUBackend 永远返回 nil error；GPU 后端可能因设备错误返回非 nil。
	Add(a, b *tensor.Tensor) (*tensor.Tensor, error)
}
