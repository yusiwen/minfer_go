// Package model 定义了模型配置和 Transformer 各层的接口。
//
// 模型配置从 GGUF 文件的 metadata 中读取，包含模型架构所需的全部参数。
// 这些参数在模型加载时确定，推理过程中不变。
package model

// Config 存储模型架构的所有超参数。
//
// 这些参数从 GGUF 文件的 metadata 中读取（如 llama.block_count、llama.attention.head_count 等）。
// 不同模型架构（Qwen2、LLaMA、Phi）的参数命名有细微差别，
// 但核心参数是相同的。
type Config struct {
	// HiddenDim 是隐藏层维度（也叫 hidden_size、n_embd）。
	// 这是 Transformer 中最核心的维度——所有中间表示都使用这个维度。
	// 常见值：Qwen2.5-0.5B = 896, Qwen2.5-1.5B = 1536, LLaMA-7B = 4096
	HiddenDim int

	// NumLayers 是 Transformer Block 的数量（也叫 n_layer）。
	// 常见值：Qwen2.5-0.5B = 24, Qwen2.5-1.5B = 28, LLaMA-7B = 32
	NumLayers int

	// NumHeads 是注意力头的数量（也叫 n_head）。
	// 每个头独立计算注意力，结果拼接后投影。
	// 常见值：Qwen2.5-0.5B = 14, Qwen2.5-1.5B = 12, LLaMA-7B = 32
	NumHeads int

	// HeadDim 是每个注意力头的维度。
	// 通常等于 HiddenDim / NumHeads。
	// 常见值：64 或 128
	HeadDim int

	// NumKVHeads 是 K 和 V 的头数（GQA/MQA 相关）。
	// MHA（默认）：NumKVHeads == NumHeads
	// GQA：NumKVHeads < NumHeads（如 LLaMA-2-70B: 8 KV heads, 64 Q heads）
	// MQA：NumKVHeads == 1（所有 Q head 共享一个 KV 投影）
	NumKVHeads int

	// FFNHiddenDim 是 FFN 中间层的维度（也叫 n_inner、intermediate_size）。
	// SwiGLU 的公式：hidden_dim × W_gate → FFNHiddenDim
	// 通常：FFNHiddenDim ≈ (8/3) * 4 * HiddenDim（考虑 SwiGLU 的 3 个投影矩阵）
	FFNHiddenDim int

	// MaxSeqLen 是模型支持的最大序列长度（上下文窗口）。
	// 常见值：Qwen2.5 = 32768, LLaMA-3 = 8192
	MaxSeqLen int

	// VocabSize 是词表大小。
	// 常见值：Qwen2.5 = 151936, LLaMA-3 = 128256
	VocabSize int

	// NormEpsilon 是归一化中的极小常数，防除零。
	// 通常为 1e-5 或 1e-6。
	NormEpsilon float32

	// RoPEBase 是 RoPE 的 base 频率。
	// 标准值为 10000.0，有些长上下文模型会增大它（如 500000.0）。
	RoPEBase float32
}

// DefaultConfig 返回一个适用于测试的默认配置。
// 这些参数模仿了一个小型模型（类似 TinyLlama）。
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
