// Package cpu implements the pure-Go CPUBackend.
//
// This is Minfer's first (and most basic) compute backend.
// All operations use plain for-loops — no SIMD (单指令多数据流), no BLAS,
// no GPU (图形处理器). The purpose:
//
//   1. Every line of math is fully visible and understandable
//      — the triple-loop matrix multiply, element-wise softmax, etc.
//   2. Serves as a correctness baseline for future optimizations
//   3. Learning value: understand the naive implementation first,
//      then understand why optimization matters
//
// Performance expectations (pure Go, single-core, float32):
//   - 0.5B model (Qwen2.5): 30-50 tok/s
//   - 3B model: 8-12 tok/s
//   - 7B model: 3-5 tok/s
//
// Optimization roadmap:
//   Phase 2: goroutine-parallel attention heads
//   Phase 3: cache-aware tiled matmul
//   Phase 4: cgo + OpenBLAS for MatMul
//   Phase 5: CUDA backend
package cpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/yusiwen/minfer/internal/compute"
	"github.com/yusiwen/minfer/internal/tensor"
)

// Compile-time check: verify CPUBackend implements compute.Backend
var _ compute.Backend = (*CPUBackend)(nil)

// CPUBackend implements the compute.Backend interface using pure Go.
// All methods use plain for-loops.
type CPUBackend struct{}

// New creates a new CPUBackend instance.
func New() *CPUBackend {
	return &CPUBackend{}
}

// MatMul computes C = A × B using the ikj triple loop with goroutine parallelism.
//
// Algorithm
// ─────────
//
// Matrix multiplication is the single most important operation in LLM (大语言模型) inference.
// A 7B model spends ~70% of its FLOPs in MatMul calls.
//
// Mathematical definition:
//   C[i][j] = Σ A[i][k] × B[k][j]   (k = 0..K-1)
//
// Triple-loop implementation (ikj order):
//   for i = 0..M-1:        // iterate over each row of A
//     for k = 0..K-1:      // for each column of A / row of B
//       aVal = A[i][k]     // stays in register for the inner loop
//       for j = 0..N-1:    // accumulate along columns of B
//         C[i][j] += aVal × B[k][j]
//
// Why ikj instead of ijk:
//
//   ijk order (naive):       ikj order (cache-friendly):
//     for i                     for i
//       for j                     for k        ← A[i][k] in register
//         for k                     for j      ← B[k][j] and C[i][j] sequential
//           ...                       ...
//
//   In the ijk order, B[k][j] access is COLUMN-MAJOR — each iteration
//   skips K elements, causing L1/L2 cache misses on every read.
//
//   In ikj order, B[k][j] access is ROW-MAJOR — sequential in memory,
//   hitting the cache on every element. A[i][k] also stays in a register
//   for the entire inner j loop (compiler hoisting), avoiding a second
//   memory load. C[i][j] is also sequential write.
//
//   Measured impact: ikj is 2-3× faster for large MatMuls (M=1, N=151936).
//
// Parallelization strategy:
//   The output C has shape [M, N]. We split the N dimension across
//   multiple goroutines so each worker computes a contiguous block
//   of C's columns.
func (b *CPUBackend) MatMul(a, bTensor *tensor.Tensor) (*tensor.Tensor, error) {
	// Shape validation
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

	// Allocate output tensor C: [M, N]
	c := tensor.New(M, N)
	numWorkers := runtime.NumCPU()
	if numWorkers > N {
		numWorkers = N
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	colsPerWorker := N / numWorkers
	var wg sync.WaitGroup

	// Fast path: Q4_0 quantized weights (dequantize on read).
	// Reduces memory bandwidth by ~4x (0.5 bytes/weight vs 4 bytes/weight).
	if bTensor.Q4Blocks != nil {
		q4 := bTensor.Q4Blocks
		const q4BlockSize = 18 // 2 bytes f16 scale + 16 bytes nibbles (32 × 4 bits)

		// Inline f16-to-f32 conversion and Q4_0 dequant.
		// Must match readQ4_0 in internal/gguf/reader.go exactly.
		f16ToF32 := func(v uint16) float32 {
			const magicDiv = float32(1.0 / (1 << 12))
			sign := uint32(v>>15) & 1
			exp := uint32(v>>10) & 0x1F
			mant := uint32(v & 0x3FF)
			if exp == 0 {
				return math.Float32frombits((sign << 31) | mant << 13) * magicDiv
			}
			if exp == 31 {
				return math.Float32frombits((sign << 31) | 0x7F800000 | mant << 13)
			}
			return math.Float32frombits((sign << 31) | ((exp - 15 + 127) << 23) | mant << 13)
		}
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			start := w * colsPerWorker
			end := start + colsPerWorker
			if w == numWorkers-1 {
				end = N
			}
			go func(colStart, colEnd int) {
				defer wg.Done()
				for i := 0; i < M; i++ {
					cRow := c.Data[i*N:]
					aRow := a.Data[i*K:]
					for k := 0; k < K; k++ {
						aVal := aRow[k]
						for j := colStart; j < colEnd; j++ {
							// Dequantize one Q4_0 value
							idx := k*N + j
							block := idx / 32
							inBlock := idx % 32
							bo := block * q4BlockSize
							// f16 scale
							scale := f16ToF32(uint16(q4[bo]) | uint16(q4[bo+1])<<8)
							// nibble extraction
							nib := q4[bo+2+inBlock/2]
							var n int8
							if inBlock%2 == 0 {
								n = int8(nib & 0x0F)
							} else {
								n = int8(nib >> 4)
							}
							bVal := (float32(n) - 8.0) * scale
							cRow[j] += aVal * bVal
						}
					}
				}
			}(start, end)
		}
		wg.Wait()
		return c, nil
	}

	// Standard path: float32 weights (ikj order, goroutine parallel)
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		start := w * colsPerWorker
		end := start + colsPerWorker
		if w == numWorkers-1 {
			end = N
		}

		go func(colStart, colEnd int) {
			defer wg.Done()

			// AVX2 fast path (decode step: M=1)
			if M == 1 {
				if TryMatmulAVX2(
					unsafe.Pointer(&a.Data[0]),
					unsafe.Pointer(&bTensor.Data[0]),
					unsafe.Pointer(&c.Data[0]),
					K, N, colStart, colEnd,
				) {
					return // AVX2 handled it
				}
			}

			// Pure Go fallback (ikj order)
			for i := 0; i < M; i++ {
				cRow := c.Data[i*N:]
				aRow := a.Data[i*K:]
				for k := 0; k < K; k++ {
					aVal := aRow[k]
					bRow := bTensor.Data[k*N:]
					for j := colStart; j < colEnd; j++ {
						cRow[j] += aVal * bRow[j]
					}
				}
			}
		}(start, end)
	}
	wg.Wait()

	return c, nil
}

// Softmax computes softmax over the last dimension (in-place).
//
// Algorithm
// ────────
//
// Softmax converts an arbitrary vector of real numbers into a probability
// distribution (all values ∈ (0,1), sum to 1).
//
// Naive version (numerically unstable — may overflow in exp):
//   Softmax(x_i) = exp(x_i) / Σ(exp(x_j))
//
// Numerically stable version (subtract max first):
//   1. Find the maximum value in the row: max_val
//   2. For each element: x_i ← exp(x_i - max_val)
//   3. Compute sum: sum = Σ(x_i)
//   4. For each element: x_i ← x_i / sum
//
// Why subtract max:
//   float32 max is ~3.4×10³⁸. exp(88) already approaches this value.
//   If any x_i is 100, exp(100) immediately overflows to +Inf.
//   After subtracting max, the largest term becomes exp(0) = 1,
//   and all terms are safely representable.
//
// Role in attention:
//   scores = Softmax(Q × K^T / √d_k)
//   After this, each row of scores tells the model
//   "how much to attend to each position in the sequence."
//
// NOTE: This implementation always returns nil. The error return exists
// for consistency with the compute.Backend interface — GPU backends may
// return non-nil on device errors.
func (b *CPUBackend) Softmax(t *tensor.Tensor) error {
	// Get the size of the last dimension
	// For [M, N] matrix: lastDim = N
	// For [B, H, S, S] attention scores: lastDim = S
	n := t.Shape[t.Dims()-1]

	// Number of "rows" to process independently
	rows := t.NumElements() / n

	for r := 0; r < rows; r++ {
		start := r * n
		end := start + n

		// Step 1: Find the row maximum (for numerical stability)
		maxVal := t.Data[start]
		for i := start + 1; i < end; i++ {
			if t.Data[i] > maxVal {
				maxVal = t.Data[i]
			}
		}

		// Step 2: Compute exp(x_i - max) and their sum
		var sum float32
		for i := start; i < end; i++ {
			t.Data[i] = float32(math.Exp(float64(t.Data[i] - maxVal)))
			sum += t.Data[i]
		}

		// Step 3: Normalize (divide by sum) → probability distribution
		for i := start; i < end; i++ {
			t.Data[i] /= sum
		}
	}

	return nil
}

// RMSNorm computes Root Mean Square Normalization (in-place).
//
// Algorithm
// ────────
//
// RMSNorm is a simplified version of Layer Normalization.
//
// LayerNorm formula:
//   y = (x - μ) / σ × γ + β
//   where μ = mean(x), σ = sqrt(var(x) + ε)
//
// RMSNorm formula (removes the mean computation):
//   y = x / RMS(x) × γ
//   where RMS(x) = sqrt( (1/N) × Σ(x_j²) + ε )
//
// By omitting the mean, RMSNorm is ~30% faster than LayerNorm,
// and experiments show it performs equally well in Transformers.
//
// The weight parameter (γ) is a learnable scaling vector applied
// element-wise — each feature dimension has its own scale factor.
func (b *CPUBackend) RMSNorm(t, weight *tensor.Tensor) error {
	// Last dimension = hidden_dim
	n := t.Shape[t.Dims()-1]
	rows := t.NumElements() / n

	// weight must be a 1-D vector with length = hidden_dim
	if weight.Dims() != 1 || weight.NumElements() != n {
		panic(fmt.Sprintf(
			"cpu.RMSNorm: weight shape mismatch: got (%d,) but input has hidden_dim %d",
			weight.NumElements(), n,
		))
	}

	// TODO: read epsilon from model.Config instead of hardcoding
	const epsilon = 1e-6 // prevents division by zero

	for r := 0; r < rows; r++ {
		start := r * n
		end := start + n

		// Step 1: Compute RMS
		// RMS = sqrt( (1/N) × Σ(x_i²) + ε )
		var sumSq float32
		for i := start; i < end; i++ {
			sumSq += t.Data[i] * t.Data[i]
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(n) + epsilon)))

		// Step 2: Normalize and multiply by learnable weight
		// y_i = x_i / RMS(x) × weight_i
		for i := start; i < end; i++ {
			t.Data[i] = (t.Data[i] / rms) * weight.Data[i-start]
		}
	}

	return nil
}

// RoPE applies Rotary Position Embedding to Q and K.
//
// Algorithm
// ────────
//
// Why is position encoding needed?
//   Self-attention is position-agnostic — swapping two tokens produces
//   the same attention scores. Position information must be injected separately.
//
// RoPE's core idea:
//   Treat each adjacent pair (q_2i, q_{2i+1}) as a vector on a 2-D plane,
//   then rotate it by an angle proportional to position pos. The rotation
//   matrix is:
//
//   [cos(pos·θ_i)  -sin(pos·θ_i)]
//   [sin(pos·θ_i)   cos(pos·θ_i)]
//
//   where θ_i = base^(-2i/d), base = 10000
//   (same base as the original Transformer sinusoidal PE)
//
// After rotation:
//   q'_2i   = q_2i × cos(θ) - q_{2i+1} × sin(θ)
//   q'_{2i+1} = q_{2i+1} × cos(θ) + q_2i × sin(θ)
//
// Frequency decay
// ──────────────
//   i=0 (earliest pair):   θ_0 = 10000⁰ = 1      → fastest rotation (short-range info)
//   i=d/2-1 (last pair):   θ ≈ 10000⁻¹ ≈ 1/10000 → slowest rotation (long-range info)
//
// This frequency decay mirrors the original Transformer sinusoidal PE:
// early dimensions encode short-range information, later ones encode long-range.
//
// In Minfer, decode phase processes one token at a time, so q and k have
// seq_len = 1.
//
// Parameters:
//   q:   Query tensor [1, num_heads, head_dim]
//   k:   Key tensor [1, num_heads, head_dim]
//   pos: Current token position in the sequence
//   dim: head_dim (from model config, e.g. 64 or 128)
func (b *CPUBackend) RoPE(q, k *tensor.Tensor, pos, dim int, base float32) error {

	// Shape validation: q and k must be 3-D [1, num_heads, dim]
	if q.Dims() != 3 || k.Dims() != 3 {
		panic("cpu.RoPE: q and k must be 3D tensors [1, num_heads, head_dim]")
	}
	if q.Shape[2] != dim || k.Shape[2] != dim {
		panic(fmt.Sprintf(
			"cpu.RoPE: head_dim mismatch: q has %d, k has %d, expected %d",
			q.Shape[2], k.Shape[2], dim,
		))
	}
	if dim%2 != 0 {
		panic(fmt.Sprintf(
			"cpu.RoPE: head_dim must be even, got %d",
			dim,
		))
	}

	qHeads := q.Shape[1]
	kHeads := k.Shape[1]
	headDim := dim

	// Apply RoPE to Q (all qHeads heads)
	for h := 0; h < qHeads; h++ {
		for i := 0; i < headDim; i += 2 {
			offset := h*headDim + i
			freq := 1.0 / math.Pow(float64(base), float64(i)/float64(headDim))
			cosVal := float32(math.Cos(float64(pos) * freq))
			sinVal := float32(math.Sin(float64(pos) * freq))
			q0 := q.Data[offset]
			q1 := q.Data[offset+1]
			q.Data[offset] = q0*cosVal - q1*sinVal
			q.Data[offset+1] = q1*cosVal + q0*sinVal
		}
	}

	// Apply RoPE to K (kHeads heads)
	for h := 0; h < kHeads; h++ {
		for i := 0; i < headDim; i += 2 {
			offset := h*headDim + i
			freq := 1.0 / math.Pow(float64(base), float64(i)/float64(headDim))
			cosVal := float32(math.Cos(float64(pos) * freq))
			sinVal := float32(math.Sin(float64(pos) * freq))
			k0 := k.Data[offset]
			k1 := k.Data[offset+1]
			k.Data[offset] = k0*cosVal - k1*sinVal
			k.Data[offset+1] = k1*cosVal + k0*sinVal
		}
	}

	return nil
}

// Silu computes the SiLU (Sigmoid Linear Unit) activation function (in-place).
//
// Mathematical definition:
//   SiLU(x) = x × sigmoid(x) = x / (1 + exp(-x))
//
// SiLU is also known as Swish (Google, 2017). It is a smooth version of ReLU:
//
//   ReLU:  x > 0 → x, x ≤ 0 → 0
//   SiLU:  small negative values on the negative side, smooth near 0
//
// The smooth gradient near 0 prevents "dead neurons" — once a ReLU neuron
// enters the negative region, its gradient is 0 forever and it stops learning.
// SiLU's gradient never completely vanishes.
//
// In the SwiGLU FFN (前馈网络):
//   gate = SiLU(x × W_gate)  ← the "gating" signal
//   up   = x × W_up           ← the "content" signal
//   out  = (gate ⊙ up) × W_down  ← gated content projected back to hidden_dim
//
// The gating mechanism lets the FFN decide which information passes through
// and which gets suppressed. This is why SwiGLU outperforms standard ReLU FFNs.
//
// NOTE: This implementation always returns nil. The error return exists
// for consistency with the compute.Backend interface — GPU backends may
// return non-nil on device errors.
func (b *CPUBackend) Silu(t *tensor.Tensor) error {
	for i := range t.Data {
		// sigmoid(x) = 1 / (1 + exp(-x))
		t.Data[i] = t.Data[i] / (1 + float32(math.Exp(-float64(t.Data[i]))))
	}
	return nil
}

// Add computes element-wise addition: c = a + b
//
// a and b must have identical shapes.
// Panics if shapes differ.
//
// In Transformers this implements the residual connection (残差连接):
//   output = sub_layer(x) + x
//
// The residual connection is crucial for training deep networks:
// without it, gradients vanish as they back-propagate through many layers.
// The residual path gives gradients a "highway" directly back to early layers.
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
