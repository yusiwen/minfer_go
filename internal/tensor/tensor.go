// Package tensor defines Minfer's core data structure: multi-dimensional tensors.
//
// A tensor represents an n-dimensional array of float32 values, stored in
// row-major order. It is the fundamental data unit in LLM (大语言模型) inference:
//   - A 0-D tensor (scalar) — e.g. a single attention score
//   - A 1-D tensor (vector) — e.g. word embedding, hidden activation
//   - A 2-D tensor (matrix) — e.g. weight matrix, Q/K/V projection
//   - A 3+ D tensor — e.g. batch attention scores (batch × head × seq_len × seq_len)
//
// During Transformer inference, all data lives in tensors and all operations
// are tensor-to-tensor. This package provides the data structure and shape manipulations.
//
// NOTE: Tensors in Minfer only handle data storage and shape transformations.
// Actual numerical computation (matrix multiply, softmax, etc.) is delegated
// to the compute.Backend interface. This design lets us add GPU backends
// without changing the Tensor type itself.
package tensor

import "fmt"

// Tensor is a multi-dimensional array.
//
// Data is stored in row-major (C-style) order in a flat []float32 slice.
// For a tensor with shape [d0, d1, d2, ..., dn], the element at index
// (i0, i1, ..., in) is located at:
//
//     offset = i0×(d1×d2×...×dn) + i1×(d2×...×dn) + ... + in
//
// Row-major means the RIGHTMOST dimension changes fastest.
//
// Example: a [2, 3] matrix
//   ⎡ a  b  c ⎤     In memory: [a, b, c, d, e, f]
//   ⎣ d  e  f ⎦     (0,0)=a, (0,1)=b, (0,2)=c, (1,0)=d, (1,1)=e, (1,2)=f
type Tensor struct {
	Data  []float32 // Flat data storage
	Shape []int     // Size of each dimension, e.g. [2, 3] = 2 rows, 3 cols

	// Q4Blocks holds raw Q4_0 quantized block data (optional).
	// When non-nil, this tensor represents quantized weights rather than
	// dequantized float32. MatMul detects this and dequantizes on the fly,
	// reducing memory bandwidth by 4x (Q4_0: 0.5 bytes/weight vs float32:
	// 4 bytes/weight).
	//
	// Layout: sequential Q4_0 blocks in row-major order. Each block holds
	// 32 values: [f16_scale (2 bytes)] [nibbles (16 bytes)].
	// Element (row, col) is at block = (row*cols + col) / 32.
	Q4Blocks []byte
}

// New creates a new tensor and allocates its memory.
// The shape parameter specifies the size of each dimension.
//
// Examples:
//   tensor.New(2, 3)     → shape [2, 3], 6 elements, zero-initialized
//   tensor.New(4)        → shape [4], 4 elements (1-D vector)
//   tensor.New(2, 3, 4)  → shape [2, 3, 4], 24 elements (3-D tensor)
//
// Panics if any dimension is negative (via memory allocation).
func New(shape ...int) *Tensor {
	// Total elements = product of all dimension sizes
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: copyShape(shape),
	}
}

// NewWithData creates a tensor from an existing data slice.
// Does NOT copy the data — it references the original slice (zero-copy).
//
// Parameters:
//   - data: existing []float32 data
//   - shape: size of each dimension
//
// Panics if the product of shape does not equal len(data).
func NewWithData(data []float32, shape ...int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	if len(data) != size {
		panic("tensor.NewWithData: data length does not match shape")
	}
	return &Tensor{
		Data:  data,
		Shape: copyShape(shape),
	}
}

// Dims returns the number of dimensions.
// Returns 0 for a scalar, 1 for a vector, 2 for a matrix, etc.
func (t *Tensor) Dims() int {
	return len(t.Shape)
}

// Size returns the size of the i-th dimension.
// Returns 1 if i is out of range (useful for scalar/broadcasting scenarios).
func (t *Tensor) Size(i int) int {
	if i < 0 || i >= len(t.Shape) {
		return 1
	}
	return t.Shape[i]
}

// NumElements returns the total number of elements (product of all dimensions).
// Equivalent to len(t.Data).
func (t *Tensor) NumElements() int {
	return len(t.Data)
}

// At returns the value at the given indices (read-only).
// Uses variadic arguments to support any number of dimensions.
//
// Examples:
//   t.At(0, 1)  → element at row 0, col 1 of a [2,3] matrix
//   t.At(3)     → 4th element of a 1-D vector
//
// Offset formula:
//   offset = Σ(i_n × stride_n), where stride is the product of remaining dimensions
//
// Panics if:
//   - Number of indices does not match the tensor's dimension count
//   - Any index is out of range for its dimension
func (t *Tensor) At(indices ...int) float32 {
	if len(indices) != len(t.Shape) {
		panic("tensor.At: index count does not match tensor dimensions")
	}
	offset := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			panic("tensor.At: index out of range")
		}
		// stride = product of dimensions after the current one
		stride := 1
		for j := i + 1; j < len(t.Shape); j++ {
			stride *= t.Shape[j]
		}
		offset += idx * stride
	}
	return t.Data[offset]
}

// Set writes a value at the given indices (mutates in place).
// Same semantics and panic conditions as At.
func (t *Tensor) Set(val float32, indices ...int) {
	if len(indices) != len(t.Shape) {
		panic("tensor.Set: index count does not match tensor dimensions")
	}
	offset := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			panic("tensor.Set: index out of range")
		}
		stride := 1
		for j := i + 1; j < len(t.Shape); j++ {
			stride *= t.Shape[j]
		}
		offset += idx * stride
	}
	t.Data[offset] = val
}

// View returns a tensor with a different shape but sharing the same underlying data.
// This is a zero-copy operation — it only changes the Shape slice, not the data.
//
// Constraint: the new shape must have the same total element count as the old one.
//
// Usage in Transformers: Views are essential for efficient reshape.
// For example, in attention computation:
//   x: [seq_len, hidden_dim]
//   View as: [seq_len, num_heads, head_dim]  — zero-cost reshape by head
//
// Panics if the new shape's element count does not match the current count.
func (t *Tensor) View(shape ...int) *Tensor {
	newSize := 1
	for _, d := range shape {
		newSize *= d
	}
	oldSize := len(t.Data)
	if newSize != oldSize {
		panic(fmt.Sprintf(
			"tensor.View: new shape has %d elements, but tensor has %d elements",
			newSize, oldSize,
		))
	}
	return &Tensor{
		Data:  t.Data,
		Shape: copyShape(shape),
	}
}

// Clone creates a deep copy of the tensor.
// Returns a brand-new Tensor with independent memory.
// Modifying the clone does NOT affect the original.
//
// When to use Clone:
//   - Saving intermediate state snapshots (e.g. residual connections)
//   - Making a modifiable copy while preserving the original
func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return &Tensor{
		Data:  data,
		Shape: copyShape(t.Shape),
	}
}

// copyShape copies a shape slice to prevent external modifications.
func copyShape(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}
