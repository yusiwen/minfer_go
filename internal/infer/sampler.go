// Package infer implements the inference engine: prefill, decode, and sampling.
//
// This package combines Phases 1-4 into a complete text generation pipeline:
//   1. Encode prompt text → token IDs (Tokenizer, Phase 3)
//   2. Prefill: process all prompt tokens in one forward pass (Model, Phase 4)
//   3. Decode: generate one token at a time using KV cache
//   4. Sample: select the next token from logits (this file)
//   5. Decode token IDs → text (Tokenizer, Phase 3)
package infer

import (
	"math"
	"math/rand"
)

// SamplerConfig controls the text generation behavior.
type SamplerConfig struct {
	Temperature float32 // >0, lower = more greedy, higher = more random
	TopK        int     // 0 = disabled; >0 = only sample from top K tokens
	TopP        float32 // 0..1, nucleus sampling threshold
}

// DefaultSamplerConfig returns sensible defaults for generation.
func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
	}
}

// Sample selects a token ID from logits using the configured strategy.
//
// The sampling pipeline:
//   1. Temperature scaling (if enabled): logits /= temperature
//   2. Apply softmax to convert logits to probabilities
//   3. Top-K filtering: keep only the K highest-probability tokens
//   4. Top-P (nucleus) filtering: keep the smallest set whose prob sum ≥ P
//   5. Sample from the filtered distribution
//
// Returns the selected token ID and its log-probability.
func Sample(logits []float32, cfg SamplerConfig) (int, float32) {
	n := len(logits)
	if n == 0 {
		return 0, 0
	}

	// Use a copy so we don't mutate the caller's logits
	probs := make([]float32, n)
	copy(probs, logits)

	// Step 1: Temperature scaling
	// Lower temperature (0 < T < 1) sharpens the distribution — high-prob tokens
	// become even more likely. Higher temperature (T > 1) flattens it —
	// all tokens become more equally likely. T=0 is greedy (always pick max).
	if cfg.Temperature > 0 && cfg.Temperature != 1.0 {
		invTemp := 1.0 / cfg.Temperature
		for i := range probs {
			probs[i] *= invTemp
		}
	}

	// Step 2: Softmax — convert logits to probabilities
	// Use numerically stable softmax (subtract max before exp)
	var maxVal float32 = probs[0]
	for _, v := range probs[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i := range probs {
		probs[i] = float32(math.Exp(float64(probs[i] - maxVal)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Step 3: Top-K filtering
	// Keep only the K highest-probability tokens. Set the rest to 0.
	if cfg.TopK > 0 && cfg.TopK < n {
		// Find the K-th highest probability
		kthProb := kthLargest(probs, cfg.TopK)
		for i := range probs {
			if probs[i] < kthProb {
				probs[i] = 0
			}
		}
		// Renormalize after filtering
		renormalize(probs)
	}

	// Step 4: Top-P (nucleus) filtering
	// Keep the smallest set of tokens whose cumulative probability ≥ P.
	if cfg.TopP > 0 && cfg.TopP < 1.0 {
		probs = topPFilter(probs, cfg.TopP)
	}

	// Step 5: Sample from the filtered distribution
	token := sampleFromProbs(probs)

	return token, float32(math.Log(float64(probs[token])))
}

// Greedy picks the token with the highest logit (temperature=0 equivalent).
// This is a separate fast path for deterministic generation.
func Greedy(logits []float32) int {
	bestIdx := 0
	bestVal := logits[0]
	for i, v := range logits[1:] {
		if v > bestVal {
			bestVal = v
			bestIdx = i + 1
		}
	}
	return bestIdx
}

// --- helpers ---

// kthLargest returns the k-th largest value in the slice (1-indexed).
// Uses selection algorithm (partial sort) — not the most efficient for large
// vocabularies but correct and simple for learning purposes.
func kthLargest(vals []float32, k int) float32 {
	if k <= 0 {
		return float32(math.Inf(1)) // all values pass
	}
	if k > len(vals) {
		return 0 // no values pass if k > vocab size
	}

	// Make a copy and sort descending
	sorted := make([]float32, len(vals))
	copy(sorted, vals)

	// Simple partial selection: find the k-th largest
	// Bubble the k largest values to the front (inefficient but simple)
	for i := 0; i < k && i < len(sorted); i++ {
		maxIdx := i
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] > sorted[maxIdx] {
				maxIdx = j
			}
		}
		sorted[i], sorted[maxIdx] = sorted[maxIdx], sorted[i]
	}

	return sorted[k-1]
}

// topPFilter keeps the smallest set of tokens whose cumulative probability ≥ P.
func topPFilter(probs []float32, p float32) []float32 {
	n := len(probs)

	// Create index-probability pairs and sort by probability descending
	type pair struct {
		idx  int
		prob float32
	}
	pairs := make([]pair, n)
	for i := range probs {
		pairs[i] = pair{i, probs[i]}
	}
	// Sort descending by probability
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if pairs[j].prob > pairs[i].prob {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Find the cutoff: smallest set whose cumulative probability ≥ p
	var cumSum float32
	cutoff := n
	for i := 0; i < n; i++ {
		cumSum += pairs[i].prob
		if cumSum >= p {
			cutoff = i + 1
			break
		}
	}

	// Build filtered probability array
	result := make([]float32, n)
	for i := 0; i < cutoff; i++ {
		result[pairs[i].idx] = pairs[i].prob
	}
	renormalize(result)

	return result
}

// sampleFromProbs samples a token index from a probability distribution.
func sampleFromProbs(probs []float32) int {
	// Compute cumulative distribution
	var total float32
	for _, p := range probs {
		total += p
	}
	if total <= 0 {
		return 0 // fallback to first token
	}

	// Generate random value in [0, total)
	r := rand.Float32() * total

	// Find which bucket it falls into
	var cum float32
	for i, p := range probs {
		cum += p
		if r <= cum {
			return i
		}
	}

	return len(probs) - 1 // fallback to last token
}

// renormalize rescales a probability slice so it sums to 1.
// If sum is 0 (all filtered away), sets first element to 1.
func renormalize(probs []float32) {
	var sum float32
	for _, v := range probs {
		sum += v
	}
	if sum <= 0 {
		if len(probs) > 0 {
			probs[0] = 1
		}
		return
	}
	for i := range probs {
		probs[i] /= sum
	}
}
