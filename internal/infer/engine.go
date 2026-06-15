package infer

import (
	"fmt"

	"github.com/yusiwen/minfer/internal/tokenizer"
)

// ModelForwarder is the minimal interface the inference engine needs
// from a model. This decouples the engine from the concrete model
// package — any model that implements Forward can be used.
type ModelForwarder interface {
	// Forward runs a forward pass. Returns per-token logits as a flat
	// []float32 (last row = logits for the next token prediction).
	Forward(tokens []int, startPos int) ([]float32, error)
}

// Engine drives the text generation loop.
//
// Pipeline:
//   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
//   │ Encode  │───►│ Prefill  │───►│ Decode   │───►│ Sample   │
//   │ text→ids│    │ full fwd │    │ 1-tok fwd│    │ logit→tok│
//   └─────────┘    └──────────┘    └──────────┘    └──────────┘
//                        │              │               │
//                   populate KV     use KV cache    collect ids
//                        │              │               │
//                        └──────────────┘               │
//                                                        ▼
//                                                  ┌──────────┐
//                                                  │ Decode   │
//                                                  │ ids→text │
//                                                  └──────────┘
type Engine struct {
	Model          ModelForwarder
	Tokenizer      tokenizer.Tokenizer
	SamplerConfig  SamplerConfig
	MaxTokens      int // maximum tokens to generate (default 512)
}

// Generate produces text from a prompt.
//
// Algorithm:
//   1. Encode prompt to token IDs
//   2. Prefill: forward all prompt tokens → populate KV cache
//   3. Sample the first generated token from logits
//   4. Decode loop: forward one token at a time, sample, append
//   5. Stop when: EOS emitted, max tokens reached
//   6. Decode generated token IDs back to text
func (e *Engine) Generate(prompt string) (string, error) {
	if e.Model == nil {
		return "", fmt.Errorf("infer: Model is nil")
	}
	if e.Tokenizer == nil {
		return "", fmt.Errorf("infer: Tokenizer is nil")
	}

	cfg := e.SamplerConfig
	if cfg.Temperature == 0 && cfg.TopK == 0 && cfg.TopP == 0 {
		cfg = DefaultSamplerConfig()
	}

	maxTokens := e.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Step 1: Encode prompt
	promptIDs, err := e.Tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("infer: encode prompt: %w", err)
	}

	// Prepend BOS if the tokenizer has one
	if bos := e.Tokenizer.BosID(); bos >= 0 && (len(promptIDs) == 0 || promptIDs[0] != bos) {
		promptIDs = append([]int{bos}, promptIDs...)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("infer: empty prompt after encoding")
	}

	// Step 2: Prefill
	logits, err := e.Model.Forward(promptIDs, 0)
	if err != nil {
		return "", fmt.Errorf("infer: prefill: %w", err)
	}

	// All generated IDs (prompt + new tokens)
	allIDs := make([]int, len(promptIDs))
	copy(allIDs, promptIDs)
	eosID := e.Tokenizer.EosID()

	// Steps 3-4: Decode loop
	for i := 0; i < maxTokens; i++ {
		// Sample next token
		var nextID int
		if cfg.Temperature > 0 {
			nextID, _ = Sample(logits, cfg)
		} else {
			nextID = Greedy(logits)
		}

		// Step 5: Check stopping conditions (check BEFORE appending)
		if eosID >= 0 && nextID == eosID {
			break
		}

		allIDs = append(allIDs, nextID)

		// Forward the single new token (decode step with KV cache)
		startPos := len(allIDs) - 1
		logits, err = e.Model.Forward([]int{nextID}, startPos)
		if err != nil {
			return "", fmt.Errorf("infer: decode step %d: %w", i, err)
		}
	}

	// Step 6: Decode generated tokens back to text
	generatedIDs := allIDs[len(promptIDs):]
	return e.Tokenizer.Decode(generatedIDs)
}
