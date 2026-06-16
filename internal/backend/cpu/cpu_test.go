package cpu

import (
	"strings"
	"testing"

	"github.com/yusiwen/minfer/internal/tensor"
)

// TestMatMul tests 2×3 × 3×2 = 2×2 matrix multiplication.
//
// Manual verification:
//   A = ⎡1 2 3⎤   B = ⎡7  8⎤
//       ⎣4 5 6⎦       ⎢9  10⎥
//                      ⎣11 12⎦
//
//   C[0][0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
//   C[0][1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
//   C[1][0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
//   C[1][1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
//
//   Result: C = ⎡58  64⎤
//               ⎣139 154⎦
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

// TestMatMul1x4x4 tests 1×4 × 4×1 = 1×1 (vector dot product).
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

// TestSoftmax verifies softmax outputs sum to 1 per row.
func TestSoftmax(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}

	// Verify all values are in (0, 1)
	for _, v := range t1.Data {
		if v <= 0 || v >= 1 {
			t.Errorf("Softmax output %f not in (0,1)", v)
		}
	}

	// Verify each row sums to 1
	if abs(t1.Data[0]+t1.Data[1]+t1.Data[2]-1) > 1e-4 {
		t.Errorf("Row 0 sum = %f, want 1", t1.Data[0]+t1.Data[1]+t1.Data[2])
	}
	if abs(t1.Data[3]+t1.Data[4]+t1.Data[5]-1) > 1e-4 {
		t.Errorf("Row 1 sum = %f, want 1", t1.Data[3]+t1.Data[4]+t1.Data[5])
	}
}

// TestAdd verifies element-wise addition.
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

// TestSiLU verifies SiLU function at known values.
func TestSiLU(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{-2, -1, 0, 1, 2}, 5)
	if err := backend.Silu(t1); err != nil {
		t.Fatal(err)
	}

	// Verify against pre-computed values
	// SiLU(x) = x * sigmoid(x)
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

	// SiLU property: there is a minimum near x ≈ -1.278 (not globally monotonic)
	// But for x ≥ 0, output should be ≥ 0
	for i := 2; i < len(t1.Data); i++ {
		if t1.Data[i] < 0 {
			t.Errorf("SiLU(%d) = %f should be >= 0 for x >= 0", i-2, t1.Data[i])
		}
	}
}

// TestRMSNorm verifies RMSNorm output has RMS = 1.
func TestRMSNorm(t *testing.T) {
	backend := New()

	// Input [1, 2, 3, 4], weight all ones
	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{1, 1, 1, 1}, 4)

	if err := backend.RMSNorm(t1, weight); err != nil {
		t.Fatal(err)
	}

	// Verify RMS of normalized vector is 1
	var sumSq float32
	for _, v := range t1.Data {
		sumSq += v * v
	}
	rms := sumSq / float32(len(t1.Data))
	if abs(rms-1) > 1e-4 {
		t.Errorf("RMS after norm = %f, want 1", rms)
	}
}

// TestRoPE verifies RoPE at position 0 (should be identity) and position 1 (should change).
func TestRoPE(t *testing.T) {
	backend := New()

	// q, k shape [1, 2, 4] (1 token, 2 heads, 4 dims)
	q := tensor.NewWithData([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 1, 2, 4)
	k := tensor.NewWithData([]float32{3, 3, 3, 3, 4, 4, 4, 4}, 1, 2, 4)

	// Save original values
	qOrig := q.Clone()
	kOrig := k.Clone()

	if err := backend.RoPE(q, k, 0, 4, 10000.0); err != nil {
		t.Fatal(err)
	}

	// Position 0: cos(0)=1, sin(0)=0, result should be unchanged
	for i := range q.Data {
		if abs(q.Data[i]-qOrig.Data[i]) > 1e-4 {
			t.Errorf("RoPE(pos=0) changed q[%d] from %f to %f", i, qOrig.Data[i], q.Data[i])
		}
		if abs(k.Data[i]-kOrig.Data[i]) > 1e-4 {
			t.Errorf("RoPE(pos=0) changed k[%d] from %f to %f", i, kOrig.Data[i], k.Data[i])
		}
	}

	// Position 1: cos≠0, sin≠0, result should differ
	if err := backend.RoPE(q, k, 1, 4, 10000.0); err != nil {
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

// Helper: absolute value for float32
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// expectPanic verifies fn panics and the panic message starts with msgPrefix.
// Verifying the message is important: it ensures the RIGHT panic fires,
// not just any panic from an unrelated code path.
func expectPanic(t *testing.T, msgPrefix string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic with prefix %q, but none occurred", msgPrefix)
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected panic string, got %T: %v", r, r)
		}
		if !strings.HasPrefix(msg, msgPrefix) {
			t.Errorf("panic message should start with %q, got %q", msgPrefix, msg)
		}
	}()
	fn()
}

// TestMatMulPanicShape verifies MatMul panics on non-2D input.
func TestMatMulPanicShape(t *testing.T) {
	backend := New()
	a := tensor.NewWithData([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	b := tensor.NewWithData([]float32{1, 2, 3, 4}, 2, 2)
	expectPanic(t, "cpu.MatMul: inputs must be 2D tensors", func() {
		backend.MatMul(a, b)
	})
}

// TestMatMulPanicInnerDim verifies MatMul panics on inner dimension mismatch.
func TestMatMulPanicInnerDim(t *testing.T) {
	backend := New()
	a := tensor.NewWithData([]float32{1, 2, 3, 4}, 2, 2)
	b := tensor.NewWithData([]float32{1, 2, 3, 4, 5, 6}, 3, 2) // A[2,2] × B[3,2] → 2≠3
	expectPanic(t, "cpu.MatMul: shape mismatch", func() {
		backend.MatMul(a, b)
	})
}

// TestSoftmaxUniform verifies uniform input → uniform output.
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

// TestSoftmaxNumericalStability verifies large values don't overflow.
//
// Without the "subtract max" trick, exp(1000) would overflow to +Inf.
// After subtracting max=1000, all terms become exp(0)=1, producing
// a uniform distribution.
func TestSoftmaxNumericalStability(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{1000, 1000, 1000}, 1, 3)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}
	for i, v := range t1.Data {
		if abs(v-1.0/3.0) > 1e-4 {
			t.Errorf("numerical stability: element %d = %f, want 0.3333", i, v)
		}
	}
}

// TestSoftmaxLargeDiff verifies behavior with extremely different values.
//
// Input [1000, 0, -1000]: max=1000, after subtract: [0, -1000, -2000]
// exp(0)=1, exp(-1000) underflows to 0, exp(-2000) underflows to 0
// Result ≈ [1, 0, 0]. The underflowed terms become exact 0 (float32 limit).
func TestSoftmaxLargeDiff(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{1000, 0, -1000}, 1, 3)
	if err := backend.Softmax(t1); err != nil {
		t.Fatal(err)
	}
	if abs(t1.Data[0]-1) > 1e-4 {
		t.Errorf("expected first element ≈ 1, got %f", t1.Data[0])
	}
	sum := t1.Data[0] + t1.Data[1] + t1.Data[2]
	if abs(sum-1) > 1e-4 {
		t.Errorf("Softmax sum = %f, want 1", sum)
	}
}

// TestRMSNorm2D verifies RMSNorm on 2-D input ([batch, dim]).
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

	// Verify each row has RMS = 1
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

	// Rows 0 and 1 should be proportional (same weight, proportional input)
	ratio := t1.Data[0] / t1.Data[4]
	for i := 0; i < 4; i++ {
		gotRatio := t1.Data[i] / t1.Data[i+4]
		if abs(gotRatio-ratio) > 1e-4 {
			t.Errorf("Row ratio mismatch at col %d: %f vs %f", i, gotRatio, ratio)
		}
	}
}

// TestRMSNormWeight verifies RMSNorm weight scaling.
func TestRMSNormWeight(t *testing.T) {
	backend := New()

	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{2, 2, 2, 2}, 4) // all 2× weight

	if err := backend.RMSNorm(t1, weight); err != nil {
		t.Fatal(err)
	}

	// Compare with unit-weight version
	t2 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	w2 := tensor.NewWithData([]float32{1, 1, 1, 1}, 4)
	backend.RMSNorm(t2, w2)

	// All-2 weight should double the output
	for i := range t1.Data {
		if abs(t1.Data[i]-t2.Data[i]*2) > 1e-4 {
			t.Errorf("Weight scaling: element %d: got %f, want %f (2×)", i, t1.Data[i], t2.Data[i]*2)
		}
	}
}

// TestRMSNormZeroInput verifies all-zero input doesn't crash (rms=epsilon, output=0).
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

// TestAddPanic verifies Add panics on shape mismatch.
func TestAddPanic(t *testing.T) {
	backend := New()
	a := tensor.NewWithData([]float32{1, 2, 3}, 3)
	b := tensor.NewWithData([]float32{4, 5, 6, 7}, 4)
	expectPanic(t, "cpu.Add: tensors have different", func() {
		backend.Add(a, b)
	})
}

// TestAddPanicDims verifies Add panics on dimension count mismatch.
func TestAddPanicDims(t *testing.T) {
	backend := New()
	a := tensor.NewWithData([]float32{1, 2, 3, 4}, 2, 2)
	b := tensor.NewWithData([]float32{4, 5, 6}, 3)
	expectPanic(t, "cpu.Add: tensors have different", func() {
		backend.Add(a, b)
	})
}

// TestRoPEPreserveNorm verifies RoPE preserves vector magnitude.
//
// RoPE is an orthogonal transformation (rotation matrix), so it must
// preserve the L2 norm. For each (2i, 2i+1) pair, the squared norm
// before and after rotation should be identical:
//   q'_2i² + q'_{2i+1}² = q_2i² + q_{2i+1}²
func TestRoPEPreserveNorm(t *testing.T) {
	backend := New()

	q := tensor.NewWithData([]float32{
		0.5, 1.5, -1.0, 2.0, // head 0: pairs (0,1), (2,3)
		1.0, -0.5, 3.0, 0.0, // head 1
	}, 1, 2, 4)

	qOrig := q.Clone()

	if err := backend.RoPE(q, q, 3, 4, 10000.0); err != nil {
		t.Fatal(err)
	}

	// For each head and each dimension pair, verify norm unchanged
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

// TestRoPEPanicDims verifies RoPE panics on non-3D input.
func TestRoPEPanicDims(t *testing.T) {
	backend := New()
	q := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	k := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	expectPanic(t, "cpu.RoPE: q and k must be 3D tensors", func() {
		backend.RoPE(q, k, 0, 4, 10000.0)
	})
}

// TestRoPEPanicDimMismatch verifies RoPE panics when dim doesn't match head_dim.
func TestRoPEPanicDimMismatch(t *testing.T) {
	backend := New()
	q := tensor.NewWithData([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 1, 2, 4)
	k := tensor.NewWithData([]float32{3, 3, 3, 3, 4, 4, 4, 4}, 1, 2, 4)
	expectPanic(t, "cpu.RoPE: head_dim mismatch", func() {
		backend.RoPE(q, k, 0, 8, 10000.0) // dim=8 ≠ q.Shape[2]=4
	})
}

// TestRoPEGQA verifies RoPE works with different Q/K head counts (GQA).
func TestRoPEGQA(t *testing.T) {
	backend := New()
	q := tensor.NewWithData([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 1, 2, 4) // 2 Q heads
	k := tensor.NewWithData([]float32{3, 3, 3, 3}, 1, 1, 4)             // 1 K head (GQA)

	qOrig := q.Clone()
	kOrig := k.Clone()

	// Should NOT panic — GQA is valid
	if err := backend.RoPE(q, k, 5, 4, 10000.0); err != nil {
		t.Fatal(err)
	}

	// Q should have changed (pos=5)
	qChanged := false
	for i := range q.Data {
		if q.Data[i] != qOrig.Data[i] {
			qChanged = true
			break
		}
	}
	if !qChanged {
		t.Error("RoPE should change Q values at pos=5")
	}

	// K should also have changed (same position)
	kChanged := false
	for i := range k.Data {
		if k.Data[i] != kOrig.Data[i] {
			kChanged = true
			break
		}
	}
	if !kChanged {
		t.Error("RoPE should change K values at pos=5")
	}
}

// TestRoPEPanicOddDim verifies RoPE panics on odd head_dim.
func TestRoPEPanicOddDim(t *testing.T) {
	backend := New()
	// head_dim=7 must equal dim=7 so the dim check passes and oddity check fires
	q := tensor.NewWithData([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 2, 7)
	k := tensor.NewWithData([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 2, 7)
	expectPanic(t, "cpu.RoPE: head_dim must be even", func() {
		backend.RoPE(q, k, 0, 7, 10000.0) // dim=7 is odd
	})
}

// TestRMSNormPanicWeightDim verifies RMSNorm panics when weight length doesn't match hidden_dim.
func TestRMSNormPanicWeightDim(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{1, 1}, 2) // hidden_dim=4, weight has 2
	expectPanic(t, "cpu.RMSNorm: weight shape mismatch", func() {
		backend.RMSNorm(t1, weight)
	})
}

// TestRMSNormPanicWeightNot1D verifies RMSNorm panics when weight is not 1-D.
func TestRMSNormPanicWeightNot1D(t *testing.T) {
	backend := New()
	t1 := tensor.NewWithData([]float32{1, 2, 3, 4}, 4)
	weight := tensor.NewWithData([]float32{1, 1, 1, 1}, 2, 2) // 2D weight
	expectPanic(t, "cpu.RMSNorm: weight shape mismatch", func() {
		backend.RMSNorm(t1, weight)
	})
}
