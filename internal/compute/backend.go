// Package compute defines Minfer's compute backend interface.
//
// This is the architectural core of the inference engine. All numerical
// computation goes through this interface rather than being called directly
// on Tensor. The design rationale:
//
//   1. Model code (internal/model/) and inference loops (internal/infer/)
//      have NO idea where computation runs — CPU, GPU (图形处理器), or other hardware.
//
//   2. Adding a new backend means implementing the Backend interface.
//      No model or inference code changes needed.
//
//   3. Learning purpose: starting with a pure-Go CPUBackend where every
//      line of math is visible, then adding SIMD (单指令多数据流) optimization,
//      CUDA, or Metal backends later.
//
// Backend selection happens at runtime via --backend flag:
//
//     minfer --backend cpu run model.gguf   → CPUBackend (default)
//     minfer --backend cuda run model.gguf  → future: CUDABackend
//     minfer --backend metal run model.gguf → future: MetalBackend
//
// Side note — what is a "kernel"?
// In HPC (高性能计算), a "kernel" is the innermost, most frequently executed
// loop of a computation — e.g. the triple loop body of matrix multiply.
// GPU programming borrowed this term: a GPU kernel is such a function
// executed in parallel on the GPU. See the Wolai page "Kernel — From OS
// to GPU Programming" for details.
package compute

import "github.com/yusiwen/minfer/internal/tensor"

// Backend is the unified interface for all compute backends.
//
// Each method corresponds to one mathematical operation in a Transformer model.
// All inputs and outputs are *tensor.Tensor.
type Backend interface {
	// MatMul computes matrix multiplication: C = A × B
	//
	// Mathematical definition:
	//   C[i][j] = Σ(A[i][k] × B[k][j]) for k = 0..K-1
	//
	// Shape constraints:
	//   A: [M, K], B: [K, N], C: [M, N]
	//
	// In Transformers, MatMul is the most frequent and expensive operation
	// (~70% of inference FLOPs):
	//   - Q = x × W_q         (hidden_dim × hidden_dim)
	//   - K = x × W_k         (hidden_dim × hidden_dim)
	//   - V = x × W_v         (hidden_dim × hidden_dim)
	//   - scores = Q × K^T    (seq_len × seq_len) — attention scores
	//   - output = scores × V (seq_len × head_dim) — weighted sum
	//   - FFN: gate = x × W_gate, up = x × W_up, down = (gate⊙up) × W_down
	//
	// CPUBackend always returns nil error; GPU backends may return error
	// on device failure (OOM, device lost, etc.).
	MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error)

	// Softmax computes softmax over the last dimension (in-place).
	//
	// Mathematical definition:
	//   Softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x))) for j = 0..N-1
	//
	// The "- max(x)" is the "numerically stable trick":
	// exp overflows float32 at around exp(88). Subtracting the global max
	// ensures the largest term becomes exp(0) = 1, keeping all values safe.
	//
	// In attention: scores = Softmax(Q × K^T / √d_k)
	// After this step, each row of scores is a probability distribution
	// (all values ∈ (0,1), each row sums to 1).
	//
	// CPUBackend always returns nil error; GPU backends may return error
	// on device failure.
	Softmax(t *tensor.Tensor) error

	// RMSNorm computes Root Mean Square Normalization (in-place).
	//
	// Mathematical definition:
	//   y_i = x_i × weight_i / RMS(x)
	//   where RMS(x) = sqrt( (1/N) × Σ(x_j²) + ε )
	//
	// Difference from LayerNorm:
	//   LayerNorm: subtract mean, divide by stddev (needs mean AND variance)
	//   RMSNorm:   only divide by RMS (skips mean, ~30% cheaper)
	//
	// RMSNorm is a simplified LayerNorm that performs equally well in practice.
	// It has completely replaced LayerNorm in LLaMA, Qwen2.5, and most modern models.
	//
	// Parameters:
	//   t:      input tensor [..., hidden_dim], modified in-place
	//   weight: learnable scaling vector [hidden_dim]
	//
	// ε (epsilon) is a tiny constant (typically 1e-6) preventing division by zero.
	RMSNorm(t, weight *tensor.Tensor) error

	// RoPE applies Rotary Position Embedding to Q and K.
	//
	// Core idea: rotate query and key vectors by an angle proportional to
	// their position, so dot products encode RELATIVE position information.
	//
	// Mathematical formula:
	//   For each position pos and each dimension pair (2i, 2i+1):
	//     θ_i = base^(-2i/d)
	//     q'_2i     = q_2i × cos(pos×θ_i) - q_{2i+1} × sin(pos×θ_i)
	//     q'_{2i+1} = q_{2i+1} × cos(pos×θ_i) + q_2i × sin(pos×θ_i)
	//     (same for K)
	//
	// Intuition:
	//   RoPE treats each adjacent pair (2i, 2i+1) as a 2-D vector and rotates
	//   it by pos×θ_i radians. The rotation frequency varies by dimension:
	//   early pairs rotate fast (encoding short-range info), later pairs rotate
	//   slowly (encoding long-range info).
	//
	// Key property (why RoPE is widely adopted):
	//   The dot product between Q at position p1 and K at position p2 depends
	//   ONLY on their relative position (p1 - p2), not absolute positions.
	//
	// Parameters:
	//   q:   Query tensor [1, num_heads, head_dim]
	//   k:   Key tensor [1, num_heads, head_dim]
	//   pos: Current token position (starting from 0)
	//   dim: head_dim (from model config, e.g. 64 or 128)
	//   base: Frequency base (e.g. 10000.0 for LLaMA, 1000000.0 for Qwen2.5)
	RoPE(q, k *tensor.Tensor, pos, dim int, base float32) error

	// Silu computes the SiLU (Sigmoid Linear Unit) activation function (in-place).
	// Also known as Swish (Google, 2017).
	//
	// Mathematical definition:
	//   SiLU(x) = x × σ(x)   where σ(x) = 1 / (1 + exp(-x))
	//
	// Shape characteristics:
	//   Negative x: small negative values (not exactly 0 like ReLU)
	//   Positive x: approximately x (slightly less)
	//   Near x=0: smooth transition (unlike ReLU's sharp corner at 0)
	//
	// Why SiLU is used in SwiGLU:
	//   FFN_out = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down
	//   The "gating" mechanism: SiLU(gate) decides how much of "up" passes through.
	//   This outperforms standard ReLU FFNs, though it adds ~1/3 more parameters.
	//
	// CPUBackend always returns nil error; GPU backends may return error
	// on device failure.
	Silu(t *tensor.Tensor) error

	// Add computes element-wise addition: c = a + b
	//
	// In Transformers this implements the residual connection:
	//   output = sub_layer(x) + x
	//
	// Residual connections prevent gradient vanishing in deep networks
	// by providing a "gradient highway" from the output back to early layers.
	//
	// CPUBackend always returns nil error; GPU backends may return error
	// on device failure.
	Add(a, b *tensor.Tensor) (*tensor.Tensor, error)
}
