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

	c := backend.MatMul(a, b)

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

	c := backend.MatMul(a, b)
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

	c := backend.Add(a, b)
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
