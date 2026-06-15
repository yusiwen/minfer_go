package tensor

import (
	"strings"
	"testing"
)

// TestNew verifies tensor creation and dimension metadata.
//
// Test targets:
//   1. New(2, 3) creates a tensor with shape [2, 3]
//   2. NumElements() returns 6 (2×3)
//   3. Dims() returns 2
//   4. Size(i) returns correct dimension sizes
func TestNew(t *testing.T) {
	t1 := New(2, 3)
	if len(t1.Shape) != 2 || t1.Shape[0] != 2 || t1.Shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", t1.Shape)
	}
	if t1.NumElements() != 6 {
		t.Fatalf("expected 6 elements, got %d", t1.NumElements())
	}
	if t1.Dims() != 2 {
		t.Fatalf("expected 2 dims, got %d", t1.Dims())
	}
	if t1.Size(0) != 2 || t1.Size(1) != 3 {
		t.Fatalf("size mismatch: got %v", t1.Shape)
	}
}

// TestNewWithData verifies creating a tensor from an existing data slice.
func TestNewWithData(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	t1 := NewWithData(data, 2, 3)
	if t1.Data[0] != 1 || t1.Data[5] != 6 {
		t.Fatalf("data mismatch: got %v", t1.Data)
	}
}

// TestAt verifies index access.
//
// For a [2,3] tensor stored row-major:
//   (0,0)→0, (0,1)→1, (0,2)→2, (1,0)→3, (1,1)→4, (1,2)→5
func TestAt(t *testing.T) {
	t1 := NewWithData([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	tests := []struct {
		i, j int
		want float32
	}{
		{0, 0, 0}, {0, 1, 1}, {0, 2, 2},
		{1, 0, 3}, {1, 1, 4}, {1, 2, 5},
	}
	for _, tt := range tests {
		got := t1.At(tt.i, tt.j)
		if got != tt.want {
			t.Errorf("At(%d,%d) = %f, want %f", tt.i, tt.j, got, tt.want)
		}
	}
}

// TestSet verifies element writes.
func TestSet(t *testing.T) {
	t1 := New(2, 3)
	t1.Set(42, 1, 1)
	if t1.Data[4] != 42 {
		t.Fatalf("Set(42, 1, 1) → Data[4] = %f, want 42", t1.Data[4])
	}
}

// TestView verifies the View operation (zero-cost reshape).
//
// Shape [2,3] → [3,2]: total elements unchanged, data shared.
func TestView(t *testing.T) {
	t1 := NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := t1.View(3, 2)

	// Shape changed
	if t2.Shape[0] != 3 || t2.Shape[1] != 2 {
		t.Fatalf("View shape: got %v, want [3,2]", t2.Shape)
	}

	// Underlying data is shared (modifying t2 affects t1)
	t2.Data[0] = 99
	if t1.Data[0] != 99 {
		t.Fatalf("View should share underlying data")
	}
}

// TestClone verifies Clone is a deep copy (data not shared).
func TestClone(t *testing.T) {
	t1 := NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := t1.Clone()

	// Modifying clone does not affect original
	t2.Data[0] = 99
	if t1.Data[0] != 1 {
		t.Fatalf("Clone should not share data: t1.Data[0] = %f, want 1", t1.Data[0])
	}
}

// Test1D verifies 1-D tensors work.
func Test1D(t *testing.T) {
	t1 := New(4)
	if t1.Dims() != 1 || t1.Shape[0] != 4 || t1.NumElements() != 4 {
		t.Fatalf("1D tensor: got shape %v, elems %d", t1.Shape, t1.NumElements())
	}
}

// Test3D verifies multi-dimensional index access for 3-D tensors.
//
// Shape [2, 2, 3] → 2 matrices, each 2×3. Row-major storage:
//   (0,0,0)=0, (0,0,1)=1, (0,0,2)=2, (0,1,0)=3, (0,1,1)=4, (0,1,2)=5
//   (1,0,0)=6, (1,0,1)=7, (1,0,2)=8, (1,1,0)=9, (1,1,1)=10, (1,1,2)=11
func Test3D(t *testing.T) {
	data := make([]float32, 12)
	for i := range data {
		data[i] = float32(i)
	}
	t1 := NewWithData(data, 2, 2, 3)

	tests := []struct {
		i, j, k int
		want    float32
	}{
		{0, 0, 0, 0}, {0, 0, 1, 1}, {0, 0, 2, 2},
		{0, 1, 0, 3}, {0, 1, 1, 4}, {0, 1, 2, 5},
		{1, 0, 0, 6}, {1, 0, 1, 7}, {1, 0, 2, 8},
		{1, 1, 0, 9}, {1, 1, 1, 10}, {1, 1, 2, 11},
	}
	for _, tt := range tests {
		got := t1.At(tt.i, tt.j, tt.k)
		if got != tt.want {
			t.Errorf("At(%d,%d,%d) = %f, want %f", tt.i, tt.j, tt.k, got, tt.want)
		}
	}
}

// TestAtPanicIndexCount verifies At panics on index count mismatch.
func TestAtPanicIndexCount(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		if msg := r.(string); !strings.HasPrefix(msg, "tensor.At: index count") {
			t.Errorf("wrong panic: %s", msg)
		}
	}()
	t1 := New(2, 3)
	t1.At(0) // 2D tensor needs 2 indices, only 1 given → panic
}

// TestAtPanicIndexOutOfRange verifies At panics on out-of-range index.
func TestAtPanicIndexOutOfRange(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		if msg := r.(string); !strings.HasPrefix(msg, "tensor.At: index out of range") {
			t.Errorf("wrong panic: %s", msg)
		}
	}()
	t1 := New(2, 3)
	t1.At(5, 0) // row 5 is out of range for shape [2,3] → panic
}

// TestSetPanicIndexCount verifies Set panics on index count mismatch.
func TestSetPanicIndexCount(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		if msg := r.(string); !strings.HasPrefix(msg, "tensor.Set: index count") {
			t.Errorf("wrong panic: %s", msg)
		}
	}()
	t1 := New(2, 3)
	t1.Set(42, 0, 0, 0) // 2D tensor needs 2 indices, 3 given → panic
}

// TestViewPanic verifies View panics on shape mismatch with informative message.
func TestViewPanic(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		msg := r.(string)
		if !strings.Contains(msg, "6") || !strings.Contains(msg, "8") {
			t.Errorf("panic message should mention element counts, got: %s", msg)
		}
	}()
	t1 := NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t1.View(4, 2) // 4×2=8 ≠ 6 → panic
}

// TestNewWithDataPanic verifies NewWithData panics on data length mismatch.
func TestNewWithDataPanic(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		msg := r.(string)
		if !strings.HasPrefix(msg, "tensor.NewWithData") {
			t.Errorf("wrong panic: %s", msg)
		}
	}()
	NewWithData([]float32{1, 2, 3}, 2, 3) // 3 elements ≠ 2×3=6 → panic
}

// TestSizeOutOfRange verifies Size returns 1 for out-of-range indices.
func TestSizeOutOfRange(t *testing.T) {
	t1 := New(2, 3)
	if t1.Size(-1) != 1 {
		t.Errorf("Size(-1) should return 1 for out-of-range")
	}
	if t1.Size(5) != 1 {
		t.Errorf("Size(5) should return 1 for out-of-range")
	}
}

// TestScalar verifies 0-D (scalar) tensor creation and access.
func TestScalar(t *testing.T) {
	t1 := New() // no args → scalar
	if t1.Dims() != 0 {
		t.Errorf("scalar tensor should have 0 dims, got %d", t1.Dims())
	}
	if t1.NumElements() != 1 {
		t.Errorf("scalar tensor should have 1 element, got %d", t1.NumElements())
	}
	// At() with no indices should return the single element
	t1.Data[0] = 42
	if t1.At() != 42 {
		t.Errorf("At() on scalar = %f, want 42", t1.At())
	}
}
