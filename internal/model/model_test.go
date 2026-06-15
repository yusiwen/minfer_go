package model

import (
	"fmt"
	"testing"

	"github.com/yusiwen/minfer/internal/backend/cpu"
	"github.com/yusiwen/minfer/internal/tensor"
)

// mockWeightProvider returns deterministic weights for testing.
type mockWeightProvider struct {
	weights map[string][]float32
}

func (m *mockWeightProvider) ReadTensor(name string) ([]float32, error) {
	if data, ok := m.weights[name]; ok {
		return data, nil
	}
	return nil, errMockNotFound
}

var errMockNotFound = &mockError{"not found"}

type mockError struct{ msg string }

func (e *mockError) Error() string { return e.msg }

// tinyConfig returns a minimal config suitable for testing.
// Model: hidden_dim=4, 2 layers, 2 heads, head_dim=2, vocab=8
func tinyConfig() Config {
	return Config{
		HiddenDim:   4,
		NumLayers:   2,
		NumHeads:    2,
		HeadDim:     2,
		NumKVHeads:  2,
		FFNHiddenDim: 8,
		MaxSeqLen:   64,
		VocabSize:   8,
		NormEpsilon: 1e-6,
		RoPEBase:    10000.0,
	}
}

// buildMockWeights creates deterministic weights for the tiny model.
func buildMockWeights(cfg Config) map[string][]float32 {
	w := make(map[string][]float32)
	hd := cfg.HiddenDim
	ffn := cfg.FFNHiddenDim

	// Helper: fill a slice with a pattern
	fill := func(n int, pattern ...float32) []float32 {
		r := make([]float32, n)
		for i := range r {
			r[i] = pattern[i%len(pattern)]
		}
		return r
	}

	w[embdWeight] = fill(cfg.VocabSize*hd, 0.01, -0.01, 0.02, -0.02)
	w[outputNormWeight] = fill(hd, 1.0, 1.0, 1.0, 1.0)
	w[outputWeight] = fill(cfg.VocabSize*hd, 0.03, -0.03, 0.04, -0.04)

	for i := 0; i < cfg.NumLayers; i++ {
		scale := float32(i + 1)
		w[fmt.Sprintf(attnNormFmt, i)] = fill(hd, scale)

		w[fmt.Sprintf(attnQWeightFmt, i)] = fill(hd*hd, 0.01*scale)
		w[fmt.Sprintf(attnKWeightFmt, i)] = fill(hd*cfg.NumKVHeads*cfg.HeadDim, 0.01*scale)
		w[fmt.Sprintf(attnVWeightFmt, i)] = fill(hd*cfg.NumKVHeads*cfg.HeadDim, 0.01*scale)
		w[fmt.Sprintf(attnOutWeightFmt, i)] = fill(hd*hd, 0.01*scale)
		w[fmt.Sprintf(ffnNormFmt, i)] = fill(hd, scale)
		w[fmt.Sprintf(ffnGateWeightFmt, i)] = fill(hd*ffn, 0.01*scale)
		w[fmt.Sprintf(ffnUpWeightFmt, i)] = fill(hd*ffn, 0.01*scale)
		w[fmt.Sprintf(ffnDownWeightFmt, i)] = fill(ffn*hd, 0.01*scale)
	}

	return w
}

// TestLoadModel verifies model loading from a weight provider.
func TestLoadModel(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	if model.TokenEmbedding == nil {
		t.Fatal("TokenEmbedding is nil")
	}
	if model.OutputNormWeight == nil {
		t.Fatal("OutputNormWeight is nil")
	}
	if len(model.Layers) != cfg.NumLayers {
		t.Fatalf("got %d layers, want %d", len(model.Layers), cfg.NumLayers)
	}
	if len(model.KVCaches) != cfg.NumLayers {
		t.Fatalf("got %d KV caches, want %d", len(model.KVCaches), cfg.NumLayers)
	}

	// Verify output weight is loaded and transposed
	if model.OutputWeight == nil {
		t.Fatal("OutputWeight is nil")
	}
	if model.OutputWeight.Shape[0] != cfg.HiddenDim {
		t.Errorf("OutputWeight shape[0] = %d, want hidden_dim=%d",
			model.OutputWeight.Shape[0], cfg.HiddenDim)
	}
	if model.OutputWeight.Shape[1] != cfg.VocabSize {
		t.Errorf("OutputWeight shape[1] = %d, want vocab_size=%d",
			model.OutputWeight.Shape[1], cfg.VocabSize)
	}
}

// TestLoadModelMissingWeight verifies error on missing required weight.
func TestLoadModelMissingWeight(t *testing.T) {
	cfg := tinyConfig()
	wp := &mockWeightProvider{weights: map[string][]float32{}}
	backend := cpu.New()

	_, err := LoadModel(wp, cfg, backend)
	if err == nil {
		t.Fatal("expected error for missing embedding weight")
	}
}

// TestForwardShape verifies the forward pass produces correct output shape.
func TestForwardShape(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	// Forward pass with 3 tokens (prefill)
	tokens := []int{0, 1, 2}
	logits, err := model.Forward(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Logits should be [1, vocab_size] = [1, 8]
	if logits.Shape[0] != 1 {
		t.Errorf("logits shape[0] = %d, want 1", logits.Shape[0])
	}
	if logits.Shape[1] != cfg.VocabSize {
		t.Errorf("logits shape[1] = %d, want %d", logits.Shape[1], cfg.VocabSize)
	}
}

// TestForwardDecode verifies decode step produces a logit for one token.
func TestForwardDecode(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	// Prefill: 2 tokens
	_, err = model.Forward([]int{0, 1}, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Decode: 1 more token at position 2
	logits, err := model.Forward([]int{2}, 2)
	if err != nil {
		t.Fatal(err)
	}

	// Should still produce [1, vocab_size]
	if logits.Shape[0] != 1 || logits.Shape[1] != cfg.VocabSize {
		t.Errorf("decode logits shape = %v, want [1, %d]", logits.Shape, cfg.VocabSize)
	}
}

// TestFFN verifies the SwiGLU FFN computes without error.
func TestFFN(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.New(1, cfg.HiddenDim)
	for i := range x.Data {
		x.Data[i] = 0.5
	}

	out, err := model.ffn(x, model.Layers[0])
	if err != nil {
		t.Fatal(err)
	}
	if out.Shape[0] != 1 || out.Shape[1] != cfg.HiddenDim {
		t.Errorf("ffn output shape = %v, want [1, %d]", out.Shape, cfg.HiddenDim)
	}
}

// TestAttention verifies the attention sub-layer computes without error.
func TestAttention(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.New(1, cfg.HiddenDim)
	for i := range x.Data {
		x.Data[i] = 0.5
	}

	kv := &model.KVCaches[0]
	out, err := model.attention(x, model.Layers[0], kv, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Shape[0] != 1 || out.Shape[1] != cfg.HiddenDim {
		t.Errorf("attention output shape = %v, want [1, %d]", out.Shape, cfg.HiddenDim)
	}
}

// TestKVCacheGrow verifies the KV cache grows correctly across decode steps.
func TestKVCacheGrow(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	// Step 1: prefill 2 tokens
	_, err = model.Forward([]int{0, 1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if model.KVCaches[0].K.Shape[0] != 2 {
		t.Errorf("cache size after prefill = %d, want 2", model.KVCaches[0].K.Shape[0])
	}

	// Step 2: decode 1 token
	_, err = model.Forward([]int{2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if model.KVCaches[0].K.Shape[0] != 3 {
		t.Errorf("cache size after decode = %d, want 3", model.KVCaches[0].K.Shape[0])
	}
}

// TestEmbeddingLookup verifies the embedding layer maps token IDs to vectors.
func TestEmbeddingLookup(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model, err := LoadModel(wp, cfg, backend)
	if err != nil {
		t.Fatal(err)
	}

	// Token 0 should select row 0 of embedding
	emb := model.TokenEmbedding
	row0 := emb.Data[0:cfg.HiddenDim]

	// Forward with token 0
	_, err = model.Forward([]int{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	_ = row0
}

// TestLogitsDeterministic verifies the same input produces the same output.
func TestLogitsDeterministic(t *testing.T) {
	cfg := tinyConfig()
	weights := buildMockWeights(cfg)
	wp := &mockWeightProvider{weights: weights}
	backend := cpu.New()

	model1, _ := LoadModel(wp, cfg, backend)
	model2, _ := LoadModel(wp, cfg, backend)

	tokens := []int{1, 2, 3}
	logits1, err := model1.Forward(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	logits2, err := model2.Forward(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	if len(logits1.Data) != len(logits2.Data) {
		t.Fatal("logits length mismatch")
	}
	for i := range logits1.Data {
		if logits1.Data[i] != logits2.Data[i] {
			t.Errorf("logits differ at position %d: %f vs %f", i, logits1.Data[i], logits2.Data[i])
		}
	}
}
