// Package model defines model configuration and Transformer layer interfaces.
//
// Model configuration is read from GGUF file metadata and contains all
// architecture hyper-parameters needed to build the model. These are
// determined at load time and never change during inference.
package model

// Config stores all hyper-parameters of the model architecture.
//
// These parameters are read from GGUF file metadata (e.g.
// llama.block_count, llama.attention.head_count). Different model
// architectures (Qwen2, LLaMA, Phi) use slightly different key names,
// but the core parameters are the same.
type Config struct {
	// HiddenDim is the hidden layer dimension (also called hidden_size, n_embd).
	// This is the central dimension of the Transformer — all intermediate
	// representations use this size.
	// Typical values: Qwen2.5-0.5B = 896, Qwen2.5-1.5B = 1536, LLaMA-7B = 4096
	HiddenDim int

	// NumLayers is the number of Transformer Blocks (also called n_layer).
	// Typical values: Qwen2.5-0.5B = 24, Qwen2.5-1.5B = 28, LLaMA-7B = 32
	NumLayers int

	// NumHeads is the number of attention heads (also called n_head).
	// Each head computes attention independently; results are concatenated
	// and projected.
	// Typical values: Qwen2.5-0.5B = 14, Qwen2.5-1.5B = 12, LLaMA-7B = 32
	NumHeads int

	// HeadDim is the dimension of each attention head.
	// Usually equals HiddenDim / NumHeads.
	// Typical values: 64 or 128
	HeadDim int

	// NumKVHeads is the number of K and V heads (for GQA/MQA).
	// MHA (Multi-Head Attention, 多头注意力): NumKVHeads == NumHeads
	// GQA (Grouped Query Attention, 分组查询注意力): NumKVHeads < NumHeads
	// MQA (Multi-Query Attention, 多查询注意力): NumKVHeads == 1
	NumKVHeads int

	// FFNHiddenDim is the intermediate dimension of the FFN (前馈网络)
	// (also called n_inner, intermediate_size).
	// SwiGLU formula: hidden_dim × W_gate → FFNHiddenDim
	// Typical: FFNHiddenDim ≈ (8/3) × 4 × HiddenDim (accounting for SwiGLU's 3 matrices)
	FFNHiddenDim int

	// MaxSeqLen is the maximum sequence length the model supports
	// (context window, 上下文窗口).
	// Typical values: Qwen2.5 = 32768, LLaMA-3 = 8192
	MaxSeqLen int

	// VocabSize is the vocabulary size (词表大小).
	// Typical values: Qwen2.5 = 151936, LLaMA-3 = 128256
	VocabSize int

	// NormEpsilon is the tiny constant in normalization to prevent division by zero.
	// Typically 1e-5 or 1e-6.
	NormEpsilon float32

	// RoPEBase is the base frequency for RoPE (旋转位置编码).
	// Standard value is 10000.0, some long-context models increase it (e.g. 500000.0).
	RoPEBase float32
}

// DefaultConfig returns a test-friendly default configuration.
// These parameters approximate a small model (similar to TinyLlama).
func DefaultConfig() Config {
	return Config{
		HiddenDim:   512,
		NumLayers:   8,
		NumHeads:    8,
		HeadDim:     64,
		NumKVHeads:  8,
		FFNHiddenDim: 2048,
		MaxSeqLen:   2048,
		VocabSize:   32000,
		NormEpsilon: 1e-6,
		RoPEBase:    10000.0,
	}
}
