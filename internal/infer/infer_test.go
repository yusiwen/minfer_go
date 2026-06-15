package infer

import (
	"math"
	"testing"
)

// mockTokenizer implements tokenizer.Tokenizer for testing.
type mockTokenizer struct {
	vocabSize int
	bosID     int
	eosID     int
	padID     int
}

func (m *mockTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}
	// Simple encoding: map each byte to its byte value as token ID
	ids := make([]int, len(text))
	for i, b := range []byte(text) {
		ids[i] = int(b) % m.vocabSize
	}
	return ids, nil
}

func (m *mockTokenizer) Decode(ids []int) (string, error) {
	bs := make([]byte, len(ids))
	for i, id := range ids {
		bs[i] = byte(id % 256)
	}
	return string(bs), nil
}

func (m *mockTokenizer) VocabSize() int { return m.vocabSize }
func (m *mockTokenizer) BosID() int     { return m.bosID }
func (m *mockTokenizer) EosID() int     { return m.eosID }
func (m *mockTokenizer) PadID() int     { return m.padID }

// mockModel implements ModelForwarder with controllable output.
type mockModel struct {
	seqLen    int
	vocabSize int
	nextToken int // token ID to return from Forward
}

func (m *mockModel) Forward(tokens []int, startPos int) ([]float32, error) {
	// Return logits: all zeros except nextToken gets a high value
	logits := make([]float32, m.vocabSize)
	if m.nextToken >= 0 && m.nextToken < m.vocabSize {
		logits[m.nextToken] = 100.0 // very high logit → always selected
	}
	return logits, nil
}

// --- Sampler tests ---

// TestGreedy verifies Greedy picks the highest logit.
func TestGreedy(t *testing.T) {
	logits := []float32{-10, -5, 0, 3, 8, -20}
	tok := Greedy(logits)
	if tok != 4 { // index 4 has value 8 (highest)
		t.Errorf("Greedy = %d, want 4 (value 8)", tok)
	}
}

// TestGreedyFirst verifies Greedy works when the first element is highest.
func TestGreedyFirst(t *testing.T) {
	logits := []float32{100, 0, -10}
	tok := Greedy(logits)
	if tok != 0 {
		t.Errorf("Greedy = %d, want 0", tok)
	}
}

// TestSampleTemperature verifies temperature changes distribution sharpness.
func TestSampleTemperatureZero(t *testing.T) {
	logits := []float32{0, 0, 100, 0} // token 2 is dominant
	cfg := SamplerConfig{Temperature: 0.1, TopK: 0, TopP: 0}

	// With very low temperature, should consistently pick token 2
	for i := 0; i < 20; i++ {
		tok, _ := Sample(logits, cfg)
		if tok != 2 {
			t.Errorf("iter %d: Sample(T=0.1) = %d, want 2", i, tok)
		}
	}
}

// TestSampleTopK verifies Top-K filtering restricts the candidate set.
func TestSampleTopK(t *testing.T) {
	// Only 2 tokens have non-zero probability after softmax
	logits := []float32{100, 0, -100, -200, -300}
	cfg := SamplerConfig{Temperature: 1.0, TopK: 2, TopP: 0}

	// Top-2 should give only tokens 0 and 1
	for i := 0; i < 50; i++ {
		tok, _ := Sample(logits, cfg)
		if tok != 0 && tok != 1 {
			t.Errorf("iter %d: Sample(TopK=2) = %d, want 0 or 1", i, tok)
		}
	}
}

// TestSampleTopKExceedsVocab verifies Top-K > vocab size doesn't crash.
func TestSampleTopKExceedsVocab(t *testing.T) {
	logits := []float32{0, 0, 0, 0}
	cfg := SamplerConfig{Temperature: 1.0, TopK: 100, TopP: 0} // K > vocab

	for i := 0; i < 10; i++ {
		tok, _ := Sample(logits, cfg)
		if tok < 0 || tok >= 4 {
			t.Errorf("iter %d: token %d out of range [0,3]", i, tok)
		}
	}
}

// TestKthLargest verifies the k-th largest selection.
func TestKthLargest(t *testing.T) {
	vals := []float32{5, 1, 9, 3, 7}
	if got := kthLargest(vals, 1); got != 9 {
		t.Errorf("1st largest = %f, want 9", got)
	}
	if got := kthLargest(vals, 3); got != 5 {
		t.Errorf("3rd largest = %f, want 5", got)
	}
	if got := kthLargest(vals, 5); got != 1 {
		t.Errorf("5th largest = %f, want 1", got)
	}
}

// TestRenormalize verifies probabilities sum to 1 after renormalization.
func TestRenormalize(t *testing.T) {
	probs := []float32{0.1, 0.3, 0.0, 0.6}
	renormalize(probs)
	var sum float32
	for _, v := range probs {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-4 {
		t.Errorf("sum after renormalize = %f, want 1", sum)
	}
}

// TestRenormalizeAllZero verifies renormalize handles all-zero probs.
func TestRenormalizeAllZero(t *testing.T) {
	probs := []float32{0, 0, 0}
	renormalize(probs)
	if probs[0] != 1 {
		t.Errorf("first element after all-zero renormalize = %f, want 1", probs[0])
	}
}

// --- Engine tests ---

// TestGenerate verifies the full generation pipeline.
func TestGenerate(t *testing.T) {
	mockTok := &mockTokenizer{vocabSize: 256, bosID: 1, eosID: 2}
	mockM := &mockModel{vocabSize: 256, nextToken: 65} // always outputs token 65 ('A')

	engine := &Engine{
		Model:         mockM,
		Tokenizer:     mockTok,
		MaxTokens:     10,
		SamplerConfig: SamplerConfig{Temperature: 0}, // greedy
	}

	text, err := engine.Generate("Hi")
	if err != nil {
		t.Fatal(err)
	}
	if len(text) == 0 {
		t.Fatal("Generate returned empty text")
	}
}

// TestGenerateStopOnEOS verifies generation stops at EOS token.
func TestGenerateStopOnEOS(t *testing.T) {
	mockTok := &mockTokenizer{vocabSize: 256, bosID: 1, eosID: 2}
	mockM := &mockModel{vocabSize: 256, nextToken: 2} // always EOS (token 2)

	engine := &Engine{
		Model:         mockM,
		Tokenizer:     mockTok,
		MaxTokens:     100, // would run a long time without EOS
		SamplerConfig: SamplerConfig{Temperature: 0},
	}

	text, err := engine.Generate("test")
	if err != nil {
		t.Fatal(err)
	}
	// Should stop immediately at EOS → no generated text
	if text != "" {
		t.Errorf("expected empty output (EOS immediately), got %q", text)
	}
}

// TestGenerateMaxTokens verifies generation stops at max tokens.
func TestGenerateMaxTokens(t *testing.T) {
	mockTok := &mockTokenizer{vocabSize: 256, bosID: -1, eosID: -1}
	mockM := &mockModel{vocabSize: 256, nextToken: 65} // always 'A'

	engine := &Engine{
		Model:         mockM,
		Tokenizer:     mockTok,
		MaxTokens:     5,
		SamplerConfig: SamplerConfig{Temperature: 0},
	}

	text, err := engine.Generate("x")
	if err != nil {
		t.Fatal(err)
	}
	if len(text) != 5 {
		t.Errorf("expected 5 A's, got %d chars: %q", len(text), text)
	}
	// Each generated token is 65 → byte 65 = 'A'
	for i, c := range text {
		if c != 'A' {
			t.Errorf("char %d = %c, want 'A'", i, c)
		}
	}
}

// TestGenerateNilModel verifies error on nil model.
func TestGenerateNilModel(t *testing.T) {
	engine := &Engine{
		Model:     nil,
		Tokenizer: &mockTokenizer{},
	}
	_, err := engine.Generate("test")
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

// TestGenerateNilTokenizer verifies error on nil tokenizer.
func TestGenerateNilTokenizer(t *testing.T) {
	engine := &Engine{
		Model:     &mockModel{},
		Tokenizer: nil,
	}
	_, err := engine.Generate("test")
	if err == nil {
		t.Fatal("expected error for nil tokenizer")
	}
}
