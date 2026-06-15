// Package model implements the Transformer neural network architecture
// for LLM (大语言模型) inference.
//
// This package connects the pieces from Phases 1-3:
//   - Tensor operations via compute.Backend (Phase 1)
//   - Model weights loaded from GGUF files (Phase 2)
//   - Tokenization via BPE/SentencePiece (Phase 3)
//
// The Model struct provides a single forward() method that processes
// token IDs through the full Transformer stack and returns logits.
package model

import (
	"fmt"
	"math"

	"github.com/yusiwen/minfer/internal/compute"
	"github.com/yusiwen/minfer/internal/tensor"
)

// --- Weight names (Qwen2.5 / LLaMA style GGUF keys) ---

const (
	embdWeight       = "token_embd.weight"
	outputWeight     = "output.weight"
	outputNormWeight = "output_norm.weight"

	attnNormFmt       = "blk.%d.attn_norm.weight"
	attnQWeightFmt    = "blk.%d.attn_q.weight"
	attnKWeightFmt    = "blk.%d.attn_k.weight"
	attnVWeightFmt    = "blk.%d.attn_v.weight"
	attnOutWeightFmt  = "blk.%d.attn_output.weight"
	ffnNormFmt        = "blk.%d.ffn_norm.weight"
	ffnGateWeightFmt  = "blk.%d.ffn_gate.weight"
	ffnUpWeightFmt    = "blk.%d.ffn_up.weight"
	ffnDownWeightFmt  = "blk.%d.ffn_down.weight"
)

// Model is the complete Transformer model, ready for inference.
//
// It holds all weight tensors and uses a compute.Backend for all
// mathematical operations. The architecture is a decoder-only
// Transformer with RoPE (旋转位置编码), RMSNorm, SwiGLU FFN (前馈网络),
// and KV cache (键值缓存).
type Model struct {
	Backend compute.Backend
	Config  Config

	// Weights
	TokenEmbedding   *tensor.Tensor // [vocab_size, hidden_dim]
	OutputWeight     *tensor.Tensor // [vocab_size, hidden_dim]
	OutputNormWeight *tensor.Tensor // [hidden_dim]

	Layers   []LayerWeights
	KVCaches []KVCache
}

// LayerWeights holds all weights for a single Transformer block.
type LayerWeights struct {
	AttnNormWeight *tensor.Tensor
	AttnQWeight    *tensor.Tensor
	AttnKWeight    *tensor.Tensor
	AttnVWeight    *tensor.Tensor
	AttnOutWeight  *tensor.Tensor
	FfnNormWeight  *tensor.Tensor
	FfnGateWeight  *tensor.Tensor
	FfnUpWeight    *tensor.Tensor
	FfnDownWeight  *tensor.Tensor
}

// KVCache stores cached Key and Value tensors from previous tokens.
// Shape: [seq_len, num_kv_heads, head_dim]
type KVCache struct {
	K *tensor.Tensor
	V *tensor.Tensor
}

// WeightProvider is the interface for loading tensors from a GGUF file.
type WeightProvider interface {
	ReadTensor(name string) ([]float32, error)
}

// LoadModel loads model weights from a GGUF file and returns a ready-to-use Model.
func LoadModel(wp WeightProvider, cfg Config, backend compute.Backend) (*Model, error) {
	m := &Model{
		Backend: backend,
		Config:  cfg,
	}

	hd := cfg.HiddenDim
	ffnHd := cfg.FFNHiddenDim

	// Embedding
	data, err := wp.ReadTensor(embdWeight)
	if err != nil {
		return nil, fmt.Errorf("loading embedding: %w", err)
	}
	m.TokenEmbedding = tensor.NewWithData(data, cfg.VocabSize, hd)

	// Output norm
	data, err = wp.ReadTensor(outputNormWeight)
	if err != nil {
		return nil, fmt.Errorf("loading output norm: %w", err)
	}
	m.OutputNormWeight = tensor.NewWithData(data, hd)

	// Output weight (LM head)
	// Stored in GGUF as [vocab_size, hidden_dim], but MatMul needs it
	// as [hidden_dim, vocab_size] since we compute: logits = hidden × W^T
	data, err = wp.ReadTensor(outputWeight)
	if err != nil {
		// Tied embedding: transpose the embedding matrix
		m.OutputWeight = transposeEmbedding(m.TokenEmbedding, cfg.VocabSize, cfg.HiddenDim)
	} else {
		m.OutputWeight = tensor.NewWithData(data, cfg.VocabSize, cfg.HiddenDim)
		// Transpose to [hidden_dim, vocab_size] for MatMul
		m.OutputWeight = transposeEmbedding(m.OutputWeight, cfg.VocabSize, cfg.HiddenDim)
	}

	// Per-layer weights
	m.Layers = make([]LayerWeights, cfg.NumLayers)
	m.KVCaches = make([]KVCache, cfg.NumLayers)

	for i := 0; i < cfg.NumLayers; i++ {
		lw := &m.Layers[i]

		read := func(name string, shape ...int) *tensor.Tensor {
			d, err := wp.ReadTensor(name)
			if err != nil {
				panic(fmt.Sprintf("layer %d: %s: %v", i, name, err))
			}
			return tensor.NewWithData(d, shape...)
		}

		lw.AttnNormWeight = read(fmt.Sprintf(attnNormFmt, i), hd)
		lw.AttnQWeight = read(fmt.Sprintf(attnQWeightFmt, i), hd, hd)
		lw.AttnKWeight = read(fmt.Sprintf(attnKWeightFmt, i), hd, cfg.NumKVHeads*cfg.HeadDim)
		lw.AttnVWeight = read(fmt.Sprintf(attnVWeightFmt, i), hd, cfg.NumKVHeads*cfg.HeadDim)
		lw.AttnOutWeight = read(fmt.Sprintf(attnOutWeightFmt, i), hd, hd)
		lw.FfnNormWeight = read(fmt.Sprintf(ffnNormFmt, i), hd)
		lw.FfnGateWeight = read(fmt.Sprintf(ffnGateWeightFmt, i), hd, ffnHd)
		lw.FfnUpWeight = read(fmt.Sprintf(ffnUpWeightFmt, i), hd, ffnHd)
		lw.FfnDownWeight = read(fmt.Sprintf(ffnDownWeightFmt, i), ffnHd, hd)

		m.KVCaches[i] = KVCache{
			K: tensor.New(0, cfg.NumKVHeads, cfg.HeadDim),
			V: tensor.New(0, cfg.NumKVHeads, cfg.HeadDim),
		}
	}

	return m, nil
}

// Forward runs one forward pass through the full Transformer model.
//
// Parameters:
//   - tokens: input token IDs [seq_len]
//   - startPos: starting position in KV cache
//
// Returns logits for the LAST token [1, vocab_size].
func (m *Model) Forward(tokens []int, startPos int) (*tensor.Tensor, error) {
	cfg := m.Config
	b := m.Backend
	seqLen := len(tokens)

	// Step 1: Embedding lookup
	input := tensor.New(seqLen, cfg.HiddenDim)
	for i, tok := range tokens {
		if tok < 0 || tok >= cfg.VocabSize {
			continue
		}
		copy(input.Data[i*cfg.HiddenDim:(i+1)*cfg.HiddenDim],
			m.TokenEmbedding.Data[tok*cfg.HiddenDim:(tok+1)*cfg.HiddenDim])
	}

	hidden := input

	// Step 2: Transformer blocks
	for i := 0; i < cfg.NumLayers; i++ {
		lw := m.Layers[i]
		kv := &m.KVCaches[i]

		// Attention sub-layer
		normed := hidden.Clone()
		if err := b.RMSNorm(normed, lw.AttnNormWeight); err != nil {
			return nil, fmt.Errorf("layer %d attn_norm: %w", i, err)
		}
		attnOut, err := m.attention(normed, lw, kv, startPos)
		if err != nil {
			return nil, fmt.Errorf("layer %d attention: %w", i, err)
		}
		hidden, err = b.Add(attnOut, hidden)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn add: %w", i, err)
		}

		// FFN sub-layer
		normed = hidden.Clone()
		if err := b.RMSNorm(normed, lw.FfnNormWeight); err != nil {
			return nil, fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}
		ffnOut, err := m.ffn(normed, lw)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}
		hidden, err = b.Add(ffnOut, hidden)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn add: %w", i, err)
		}
	}

	// Step 3: Final RMSNorm on last token
	last := tensor.NewWithData(
		hidden.Data[(seqLen-1)*cfg.HiddenDim:seqLen*cfg.HiddenDim],
		cfg.HiddenDim,
	)
	if err := b.RMSNorm(last, m.OutputNormWeight); err != nil {
		return nil, fmt.Errorf("output_norm: %w", err)
	}

	// Step 4: LM head projection
	logits, err := b.MatMul(last.View(1, cfg.HiddenDim), m.OutputWeight)
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}
	return logits, nil
}

// attention computes multi-head attention with RoPE and KV cache.
func (m *Model) attention(x *tensor.Tensor, lw LayerWeights, kv *KVCache, startPos int) (*tensor.Tensor, error) {
	cfg := m.Config
	b := m.Backend
	seqLen := x.Shape[0]
	headDim := cfg.HeadDim

	// Project Q, K, V
	q, err := b.MatMul(x, lw.AttnQWeight)
	if err != nil {
		return nil, err
	}
	k, err := b.MatMul(x, lw.AttnKWeight)
	if err != nil {
		return nil, err
	}
	v, err := b.MatMul(x, lw.AttnVWeight)
	if err != nil {
		return nil, err
	}

	q = q.View(seqLen, cfg.NumHeads, headDim)
	k = k.View(seqLen, cfg.NumKVHeads, headDim)
	v = v.View(seqLen, cfg.NumKVHeads, headDim)

	// Apply RoPE per position
	tmpQ := tensor.New(1, cfg.NumHeads, headDim)
	tmpK := tensor.New(1, cfg.NumKVHeads, headDim)
	for pos := 0; pos < seqLen; pos++ {
		qOff := pos * cfg.NumHeads * headDim
		kOff := pos * cfg.NumKVHeads * headDim
		copy(tmpQ.Data, q.Data[qOff:qOff+cfg.NumHeads*headDim])
		copy(tmpK.Data, k.Data[kOff:kOff+cfg.NumKVHeads*headDim])
		if err := b.RoPE(tmpQ, tmpK, startPos+pos, headDim); err != nil {
			return nil, err
		}
		copy(q.Data[qOff:qOff+cfg.NumHeads*headDim], tmpQ.Data)
		copy(k.Data[kOff:kOff+cfg.NumKVHeads*headDim], tmpK.Data)
	}

	// Extend KV cache
	cacheLen := startPos + seqLen
	newK := tensor.New(cacheLen, cfg.NumKVHeads, headDim)
	newV := tensor.New(cacheLen, cfg.NumKVHeads, headDim)
	if startPos > 0 {
		copy(newK.Data, kv.K.Data[:startPos*cfg.NumKVHeads*headDim])
		copy(newV.Data, kv.V.Data[:startPos*cfg.NumKVHeads*headDim])
	}
	copy(newK.Data[startPos*cfg.NumKVHeads*headDim:], k.Data)
	copy(newV.Data[startPos*cfg.NumKVHeads*headDim:], v.Data)
	kv.K = newK
	kv.V = newV

	// Per-head attention
	output := tensor.New(seqLen, cfg.NumHeads, headDim)

	for h := 0; h < cfg.NumHeads; h++ {
		kvH := h
		if cfg.NumKVHeads < cfg.NumHeads {
			kvH = h * cfg.NumKVHeads / cfg.NumHeads
		}

		for qi := 0; qi < seqLen; qi++ {
			qBase := qi*cfg.NumHeads*headDim + h*headDim
			maxPos := qi + startPos // causal: can only attend up to current position
			if maxPos >= cacheLen {
				maxPos = cacheLen - 1
			}
			nScores := maxPos + 1

			// Compute softmax scores
			scores := make([]float32, nScores)
			var maxScore float32
			first := true

			for ki := 0; ki < nScores; ki++ {
				kBase := ki*cfg.NumKVHeads*headDim + kvH*headDim
				var s float32
				for d := 0; d < headDim; d++ {
					s += q.Data[qBase+d] * kv.K.Data[kBase+d]
				}
				s /= float32(headDim)
				scores[ki] = s
				if first || s > maxScore {
					maxScore = s
					first = false
				}
			}

			// Numerically stable softmax
			var sum float32
			for ki := 0; ki < nScores; ki++ {
				scores[ki] = float32(math.Exp(float64(scores[ki] - maxScore)))
				sum += scores[ki]
			}

			// Weighted sum of V
			dstOff := qi*cfg.NumHeads*headDim + h*headDim
			for d := 0; d < headDim; d++ {
				var val float32
				for ki := 0; ki < nScores; ki++ {
					vBase := ki*cfg.NumKVHeads*headDim + kvH*headDim
					val += (scores[ki] / sum) * kv.V.Data[vBase+d]
				}
				output.Data[dstOff+d] = val
			}
		}
	}

	// Project back to hidden_dim
	output = output.View(seqLen, cfg.NumHeads*headDim)
	return b.MatMul(output, lw.AttnOutWeight)
}

// ffn computes SwiGLU FFN: SiLU(x×W_gate) ⊙ (x×W_up) × W_down
func (m *Model) ffn(x *tensor.Tensor, lw LayerWeights) (*tensor.Tensor, error) {
	b := m.Backend

	gate, err := b.MatMul(x, lw.FfnGateWeight)
	if err != nil {
		return nil, err
	}
	if err := b.Silu(gate); err != nil {
		return nil, err
	}

	up, err := b.MatMul(x, lw.FfnUpWeight)
	if err != nil {
		return nil, err
	}

	if len(gate.Data) != len(up.Data) {
		return nil, fmt.Errorf("ffn: gate/up size mismatch: %d vs %d", len(gate.Data), len(up.Data))
	}
	for i := range gate.Data {
		gate.Data[i] *= up.Data[i]
	}

	return b.MatMul(gate, lw.FfnDownWeight)
}

// transposeEmbedding transposes a [vocab_size, hidden_dim] embedding matrix
// to [hidden_dim, vocab_size] for use as an LM head in MatMul.
func transposeEmbedding(emb *tensor.Tensor, vocabSize, hiddenDim int) *tensor.Tensor {
	data := make([]float32, vocabSize*hiddenDim)
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < hiddenDim; j++ {
			data[j*vocabSize+i] = emb.Data[i*hiddenDim+j]
		}
	}
	return tensor.NewWithData(data, hiddenDim, vocabSize)
}
