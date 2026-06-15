package tensor

import (
	"testing"
)

// TestNew 验证张量创建和维度信息。
//
// 测试目标：
//   1. New(2, 3) 创建形状 [2, 3] 的张量
//   2. NumElements() 返回 6（2×3）
//   3. Dims() 返回 2
//   4. Size(i) 返回正确维度大小
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

// TestNewWithData 验证从数据切片创建张量。
func TestNewWithData(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	t1 := NewWithData(data, 2, 3)
	if t1.Data[0] != 1 || t1.Data[5] != 6 {
		t.Fatalf("data mismatch: got %v", t1.Data)
	}
}

// TestAt 验证索引访问。
//
// 行优先存储下，形状 [2,3] 的张量：
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

// TestSet 验证写入元素。
func TestSet(t *testing.T) {
	t1 := New(2, 3)
	t1.Set(42, 1, 1)
	if t1.Data[4] != 42 {
		t.Fatalf("Set(42, 1, 1) → Data[4] = %f, want 42", t1.Data[4])
	}
}

// TestView 验证 View 操作（零成本 reshape）。
//
// 形状 [2,3] → [3,2]，总元素数不变，共享底层数据。
func TestView(t *testing.T) {
	t1 := NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := t1.View(3, 2)

	// 形状改变
	if t2.Shape[0] != 3 || t2.Shape[1] != 2 {
		t.Fatalf("View shape: got %v, want [3,2]", t2.Shape)
	}

	// 共享底层数据（修改 t2 影响 t1）
	t2.Data[0] = 99
	if t1.Data[0] != 99 {
		t.Fatalf("View should share underlying data")
	}
}

// TestClone 验证 Clone 是深度拷贝（不共享数据）。
func TestClone(t *testing.T) {
	t1 := NewWithData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := t1.Clone()

	// 修改克隆不影响原张量
	t2.Data[0] = 99
	if t1.Data[0] != 1 {
		t.Fatalf("Clone should not share data: t1.Data[0] = %f, want 1", t1.Data[0])
	}
}

// Test1D 验证 1 维张量也能工作。
func Test1D(t *testing.T) {
	t1 := New(4)
	if t1.Dims() != 1 || t1.Shape[0] != 4 || t1.NumElements() != 4 {
		t.Fatalf("1D tensor: got shape %v, elems %d", t1.Shape, t1.NumElements())
	}
}
