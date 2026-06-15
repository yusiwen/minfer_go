package cpu

import (
	"testing"

	"github.com/yusiwen/minfer/internal/tensor"
)

// TestMatMul 测试 2×3 × 3×2 = 2×2 的矩阵乘法。
//
// 手工验证：
//   A = ⎡1 2 3⎤   B = ⎡7  8⎤
//       ⎣4 5 6⎦       ⎢9  10⎥
//                      ⎣11 12⎦
//
//   C[0][0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
//   C[0][1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
//   C[1][0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
//   C[1][1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
//
//   结果: C = ⎡58  64⎤
//            ⎣139 154⎦
func TestMatMul(t *testing.T) {
	backend := New()

	a := tensor.NewWithData([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)

	b := tensor.NewWithData([]float32{
		7, 8,
		9, 10,
		11, 12,
	}, 3, 2)

	c, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float32{58, 64, 139, 154}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("C[%d] = %f, want %f", i, c.Data[i], v)
		}
	}
}

// TestMatMul1x4x4 测试 1×4 × 4×1 = 1×1（向量内积）。
func TestMatMul1x4x4(t *testing.T) {
	backend := New()

	a := tensor.NewWithData([]float32{1, 2, 3, 4}, 1, 4)
	b := tensor.NewWithData([]float32{5, 6, 7, 8}, 4, 1)

	c, err := backend.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	expected := float32(1*5 + 2*6 + 3*7 + 4*8) // 70
	if c.Data[0] != expected {
		t.Errorf("dot product = %f, want %f", c.Data[0], expected)
	}
}

// TestSoftmax 验证 Softmax 结果行和为 1。
func TestSoftmax(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}

	// 验证所有值都在 (0, 1) 之间
	for _, v := range t1.Data {
		if v <= 0 || v >= 1 {
			t.Errorf("Softmax output %f not in (0,1)", v)
		}
	}

	// 验证每行和为 1
	if abs(t1.Data[0]+t1.Data[1]+t1.Data[2]-1) > 1e-4 {
		t.Errorf("Row 0 sum = %f, want 1", t1.Data[0]+t1.Data[1]+t1.Data[2])
	}
	if abs(t1.Data[3]+t1.Data[4]+t1.Data[5]-1) > 1e-4 {
		t.Errorf("Row 1 sum = %f, want 1", t1.Data[3]+t1.Data[4]+t1.Data[5])
	}
}

// TestAdd 验证逐元素加法。
func TestAdd(t *testing.T) {
	backend := New()

	a := tensor.NewWithData([]float32{1, 2, 3}, 3)
	b := tensor.NewWithData([]float32{4, 5, 6}, 3)

	c, err := backend.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{5, 7, 9}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("Add[%d] = %f, want %f", i, c.Data[i], v)
		}
	}
}

// TestSiLU 验证 SiLU 函数的边界值。
func TestSiLU(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{-2, -1, 0, 1, 2}, 5)
	if err := backend.Silu(t1); err != nil {
		t.Fatal(err)
	}

	// 验证已知值（使用高精度计算工具预先算好的）
	// SiLU(x) = x * sigmoid(x)，手工验证几个点
	expected := []float32{
		-2 * 0.1192029, // SiLU(-2) ≈ -0.238
		-1 * 0.2689414, // SiLU(-1) ≈ -0.269
		0 * 0.5,        // SiLU(0) = 0
		1 * 0.7310586,  // SiLU(1) ≈ 0.731
		2 * 0.8807971,  // SiLU(2) ≈ 1.762
	}
	for i, exp := range expected {
		if abs(t1.Data[i]-exp) > 1e-3 {
			t.Errorf("SiLU[%d] = %f, want ~%f (input was %d)", i, t1.Data[i], exp, i-2)
		}
	}

	// SiLU 的性质：存在一个最小值（约 x ≈ -1.278），不是全局单调的
	// 但 SiLU(0)=0，且 x>0 时 SiLU(x)>0
	for i := 2; i < len(t1.Data); i++ {
		if t1.Data[i] < 0 {
			t.Errorf("SiLU(%d) = %f should be >= 0 for x >= 0", i-2, t1.Data[i])
		}
	}
}

// TestRMSNorm 验证 RMSNorm 的 RMS 值。
func TestRMSNorm(t *testing.T) {
	backend := New()

	// 输入 [1, 2, 3, 4]，weight 全 1
	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{1, 1, 1, 1}, 4)

	if err := backend.RMSNorm(t1, weight); err != nil {
		t.Fatal(err)
	}

	// 验证 RMSNorm 后向量的 RMS 为 1
	var sumSq float32
	for _, v := range t1.Data {
		sumSq += v * v
	}
	rms := sumSq / float32(len(t1.Data))
	if abs(rms-1) > 1e-4 {
		t.Errorf("RMS after norm = %f, want 1", rms)
	}
}

// TestRoPE 验证 RoPE 的结果维度不变。
func TestRoPE(t *testing.T) {
	backend := New()

	// q, k 形状 [1, 2, 4]（1 token, 2 heads, 4 dims）
	q := tensor.NewWithData([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 1, 2, 4)
	k := tensor.NewWithData([]float32{3, 3, 3, 3, 4, 4, 4, 4}, 1, 2, 4)

	// 保存原始数据
	qOrig := q.Clone()
	kOrig := k.Clone()

	if err := backend.RoPE(q, k, 0, 4); err != nil {
		t.Fatal(err)
	}

	// 验证位置 0：cos(0)=1, sin(0)=0，结果应不变
	for i := range q.Data {
		if abs(q.Data[i]-qOrig.Data[i]) > 1e-4 {
			t.Errorf("RoPE(pos=0) changed q[%d] from %f to %f", i, qOrig.Data[i], q.Data[i])
		}
		if abs(k.Data[i]-kOrig.Data[i]) > 1e-4 {
			t.Errorf("RoPE(pos=0) changed k[%d] from %f to %f", i, kOrig.Data[i], k.Data[i])
		}
	}

	// 验证位置 1：cos≠0, sin≠0，结果应变化
	if err := backend.RoPE(q, k, 1, 4); err != nil {
		t.Fatal(err)
	}
	same := true
	for i := range q.Data {
		if q.Data[i] != qOrig.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("RoPE(pos=1) should change the values")
	}
}

// 辅助函数
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// TestMatMulPanicShape 验证 MatMul 在输入不是 2D 时 panic。
func TestMatMulPanicShape(t *testing.T) {
	backend := New()

	// 3D tensor
	a := tensor.NewWithData([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	b := tensor.NewWithData([]float32{1, 2, 3, 4}, 2, 2)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 3D input")
		}
	}()
	backend.MatMul(a, b)
}

// TestMatMulPanicInnerDim 验证 MatMul 在内维度不匹配时 panic。
func TestMatMulPanicInnerDim(t *testing.T) {
	backend := New()

	a := tensor.NewWithData([]float32{1, 2, 3, 4}, 2, 2)
	b := tensor.NewWithData([]float32{1, 2, 3, 4, 5, 6}, 3, 2) // A[2,2] × B[3,2] → 2≠3

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for inner dim mismatch")
		}
	}()
	backend.MatMul(a, b)
}

// TestSoftmaxUniform 验证所有值相等时，Softmax 输出均匀分布。
func TestSoftmaxUniform(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{5, 5, 5, 5}, 1, 4)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}
	for _, v := range t1.Data {
		if abs(v-0.25) > 1e-4 {
			t.Errorf("uniform Softmax: got %f, want 0.25", v)
		}
	}
}

// TestSoftmaxNumericalStability 验证大值不会溢出。
//
// 如果没有数值稳定技巧（先减 max），exp(1000) 会 overflow 到 +Inf。
// 减去 max=1000 后，所有项变成 exp(0)=1，结果应该是均匀分布。
func TestSoftmaxNumericalStability(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{1000, 1000, 1000}, 1, 3)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}
	// 如果没有溢出，结果就是 1/3 ≈ 0.3333
	for i, v := range t1.Data {
		if abs(v-1.0/3.0) > 1e-4 {
			t.Errorf("numerical stability: element %d = %f, want 0.3333", i, v)
		}
	}
}

// TestSoftmaxLargeDiff 验证值差距极大时 exp 不会下溢成 0。
//
// 输入 [1000, 0, -1000] → max=1000 减 max 后变成 [0, -1000, -2000]
// exp(0)=1, exp(-1000)=0 (underflow), exp(-2000)=0 (underflow)
// 结果应该接近 [1, 0, 0]，但下溢的项变成严格 0 而非极小正数。
// 这是 float32 精度限制，在 LLM 推理中不是问题。
func TestSoftmaxLargeDiff(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{1000, 0, -1000}, 1, 3)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}
	// 第一个元素 ≈ 1，后两个接近 0
	if abs(t1.Data[0]-1) > 1e-4 {
		t.Errorf("expected first element ≈ 1, got %f", t1.Data[0])
	}
	// 和为 1
	sum := t1.Data[0] + t1.Data[1] + t1.Data[2]
	if abs(sum-1) > 1e-4 {
		t.Errorf("Softmax sum = %f, want 1", sum)
	}
}

// TestRMSNorm2D 验证 2 维输入（[batch, dim]）的 RMSNorm。
func TestRMSNorm2D(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{
		1, 2, 3, 4,
		2, 4, 6, 8,
	}, 2, 4)
	weight := tensor.NewWithData([]float32{1, 1, 1, 1}, 4)

	if err := backend.RMSNorm(t1, weight); err != nil {
		t.Fatal(err)
	}

	// 验证每行的 RMS 为 1
	for r := 0; r < 2; r++ {
		start := r * 4
		var sumSq float32
		for i := start; i < start+4; i++ {
			sumSq += t1.Data[i] * t1.Data[i]
		}
		rms := sumSq / 4
		if abs(rms-1) > 1e-4 {
			t.Errorf("Row %d RMS = %f, want 1 (after RMSNorm)", r, rms)
		}
	}

	// 第 0 行和第 1 行的形状应该成比例（weight 相同，输入成比例）
	ratio := t1.Data[0] / t1.Data[4]
	for i := 0; i < 4; i++ {
		gotRatio := t1.Data[i] / t1.Data[i+4]
		if abs(gotRatio-ratio) > 1e-4 {
			t.Errorf("Row ratio mismatch at col %d: %f vs %f", i, gotRatio, ratio)
		}
	}
}

// TestRMSNormWeight 验证 RMSNorm 的 weight 缩放效果。
func TestRMSNormWeight(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{2, 2, 2, 2}, 4) // 全 2 倍的 weight

	if err := backend.RMSNorm(t1, weight); err != nil {
		t.Fatal(err)
	}

	// 用全 1 weight 的版本做对比
	t2 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	w2 := tensor.NewWithData([]float32{1, 1, 1, 1}, 4)
	backend.RMSNorm(t2, w2)

	// 全 2 的 weight 应该使输出翻倍
	for i := range t1.Data {
		if abs(t1.Data[i]-t2.Data[i]*2) > 1e-4 {
			t.Errorf("Weight scaling: element %d: got %f, want %f (2×)", i, t1.Data[i], t2.Data[i]*2)
		}
	}
}

// TestRMSNormZeroInput 验证全零输入不会出错（rms=epsilon，输出为零）。
func TestRMSNormZeroInput(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{0, 0, 0, 0}, 4)
	weight := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)

	if err := backend.RMSNorm(t1, weight); err != nil {
		t.Fatal(err)
	}

	for i, v := range t1.Data {
		if v != 0 {
			t.Errorf("Zero input: element %d = %f, want 0", i, v)
		}
	}
}

// TestAddPanic 验证 Add 在形状不匹配时 panic。
func TestAddPanic(t *testing.T) {
	backend := New()

	a := tensor.NewWithData([]float32{1, 2, 3}, 3)
	b := tensor.NewWithData([]float32{4, 5, 6, 7}, 4)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for shape mismatch")
		}
	}()
	backend.Add(a, b)
}

// TestAddPanicDims 验证 Add 在维度数不匹配时 panic。
func TestAddPanicDims(t *testing.T) {
	backend := New()

	a := tensor.NewWithData([]float32{1, 2, 3, 4}, 2, 2)
	b := tensor.NewWithData([]float32{4, 5, 6}, 3)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim mismatch")
		}
	}()
	backend.Add(a, b)
}

// TestRoPEPreserveNorm 验证 RoPE 保持向量模长。
//
// RoPE 是正交变换（旋转矩阵），不改变向量长度。
// 对每个 (2i, 2i+1) 对，旋转前后模长应该不变：
//   q'_2i² + q'_{2i+1}² = q_2i² + q_{2i+1}²
func TestRoPEPreserveNorm(t *testing.T) {
	backend := New()

	// 使用非平凡的初始值
	q := tensor.NewWithData([]float32{
		0.5, 1.5, -1.0, 2.0, // head 0: (0,1), (2,3)
		1.0, -0.5, 3.0, 0.0, // head 1
	}, 1, 2, 4)

	qOrig := q.Clone()

	if err := backend.RoPE(q, q, 3, 4); err != nil {
		t.Fatal(err)
	}

	// 对每个 head 和每个维度对，验证模长不变
	headDim := 4
	numHeads := 2
	for h := 0; h < numHeads; h++ {
		for i := 0; i < headDim; i += 2 {
			off := h*headDim + i
			origNorm := qOrig.Data[off]*qOrig.Data[off] + qOrig.Data[off+1]*qOrig.Data[off+1]
			newNorm := q.Data[off]*q.Data[off] + q.Data[off+1]*q.Data[off+1]
			if abs(newNorm-origNorm) > 1e-4 {
				t.Errorf("Head %d, pair (%d,%d): norm changed from %f to %f",
					h, i, i+1, origNorm, newNorm)
			}
		}
	}
}

// TestRoPEPanicDims 验证 RoPE 在输入不是 3D 时 panic。
func TestRoPEPanicDims(t *testing.T) {
	backend := New()

	q := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)    // 1D
	k := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)    // 1D

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 1D input")
		}
	}()
	backend.RoPE(q, k, 0, 4)
}

// TestRoPEPanicDimMismatch 验证 RoPE 在 dim 与 q/k 的 head_dim 不匹配时 panic。
func TestRoPEPanicDimMismatch(t *testing.T) {
	backend := New()

	q := tensor.NewWithData([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 1, 2, 4)
	k := tensor.NewWithData([]float32{3, 3, 3, 3, 4, 4, 4, 4}, 1, 2, 4)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim mismatch")
		}
	}()
	backend.RoPE(q, k, 0, 8) // dim=8 ≠ q.Shape[2]=4 → panic
}

// TestRoPEPanicHeadsMismatch 验证 RoPE 在 q/k 的 num_heads 不同时 panic。
func TestRoPEPanicHeadsMismatch(t *testing.T) {
	backend := New()

	q := tensor.NewWithData([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 1, 2, 4)
	k := tensor.NewWithData([]float32{3, 3, 3, 3}, 1, 1, 4) // 1 head vs 2 heads

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for num_heads mismatch")
		}
	}()
	backend.RoPE(q, k, 0, 4)
}

// TestRoPEPanicOddDim 验证 RoPE 在 head_dim 为奇数时 panic。
func TestRoPEPanicOddDim(t *testing.T) {
	backend := New()

	q := tensor.NewWithData([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 2, 6)
	k := tensor.NewWithData([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 2, 6)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for odd head_dim")
		}
	}()
	backend.RoPE(q, k, 0, 7) // dim=7 is odd → panic
}

// TestRMSNormPanicWeightDim 验证 RMSNorm 在 weight 维度不匹配时 panic。
func TestRMSNormPanicWeightDim(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{1, 1}, 2) // hidden_dim=4, weight has 2

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for weight dim mismatch")
		}
	}()
	backend.RMSNorm(t1, weight)
}

// TestRMSNormPanicWeightNot1D 验证 RMSNorm 在 weight 不是1D时 panic。
func TestRMSNormPanicWeightNot1D(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{1, 1, 1, 1}, 2, 2) // 2D weight

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 2D weight")
		}
	}()
	backend.RMSNorm(t1, weight)
}
