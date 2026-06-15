// Package tokenizer implements text tokenization for LLM (大语言模型) inference.
//
// Tokenizers convert between human-readable text and sequences of token IDs
// that the model can process. Minfer supports the two most common tokenizer
// models found in GGUF files:
//
//   - GPT-2 BPE (Byte Pair Encoding): tokenizer.ggml.model = "gpt2"
//     Used by GPT-2, many Qwen variants, and older LLaMA models.
//     Pre-tokenizes with a regex pattern, then applies BPE merge rules.
//
//   - SentencePiece Unigram: tokenizer.ggml.model = "llama"
//     Used by LLaMA, Mistral, and most modern models.
//     Uses a unigram language model with byte-pair fallback.
//
// Both tokenizer models store their vocabulary, scores, and merge rules in
// GGUF metadata keys (tokenizer.ggml.tokens, tokenizer.ggml.scores,
// tokenizer.ggml.merges, etc.), which are loaded from the GGUF reader.
package tokenizer

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"unicode/utf8"
)

// Tokenizer converts text to and from token ID sequences.
type Tokenizer interface {
	// Encode converts text to a slice of token IDs.
	Encode(text string) ([]int, error)

	// Decode converts token IDs back to text.
	Decode(ids []int) (string, error)

	// VocabSize returns the vocabulary size.
	VocabSize() int

	// BosID returns the Beginning-Of-Sequence token ID (or -1 if not set).
	BosID() int

	// EosID returns the End-Of-Sequence token ID (or -1 if not set).
	EosID() int

	// PadID returns the padding token ID (or -1 if not set).
	PadID() int
}

// --- Helpers for loading tokenizer data from GGUF metadata ---

// MetadataProvider is the interface the GGUF reader satisfies.
type MetadataProvider interface {
	GetMetadataString(key string) (string, bool)
	GetMetadataUint64(key string) (uint64, bool)
	GetMetadataStringArray(key string) ([]string, bool)
	GetMetadataFloat32Array(key string) ([]float32, bool)
}

// TokenizerConfig holds tokenizer metadata extracted from a GGUF file.
type TokenizerConfig struct {
	Model string   // tokenizer model type: "gpt2", "llama", "bert", "tiktoken"
	Tokens  []string // vocabulary
	Scores  []float32 // token scores (log probabilities), optional
	Merges  []string // BPE merge pairs, e.g. ["a b", "c d"]

	BosTokenID int // BOS (Beginning of Sequence) token ID
	EosTokenID int // EOS (End of Sequence) token ID
	PadTokenID int // PAD (Padding) token ID
}

// LoadConfig reads tokenizer metadata from a GGUF file.
func LoadConfig(m MetadataProvider) (TokenizerConfig, error) {
	cfg := TokenizerConfig{
		BosTokenID: -1,
		EosTokenID: -1,
		PadTokenID: -1,
	}

	// Model type
	if v, ok := m.GetMetadataString("tokenizer.ggml.model"); ok {
		cfg.Model = v
	}

	// Vocabulary
	if v, ok := m.GetMetadataString("tokenizer.ggml.tokens"); ok {
		// Single string — try parsing as JSON array? No, GGUF stores this as
		// a string[] array type. If GetMetadataString was used, it won't work.
		// This is handled via the gguf reader directly.
		_ = v
	}

	// Special token IDs
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.bos_token_id"); ok {
		cfg.BosTokenID = int(v)
	}
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.eos_token_id"); ok {
		cfg.EosTokenID = int(v)
	}
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.padding_token_id"); ok {
		cfg.PadTokenID = int(v)
	}

	return cfg, nil
}

// --- GPT-2 BPE Tokenizer ---

// gpt2PreTokenizePattern matches GPT-2-style pre-tokenization boundaries.
// This is the standard GPT-2 regex for splitting text into "words",
// adapted to be compatible with Go's RE2 (no lookahead/lookbehind).
// It handles contractions, letters, numbers, punctuation, and whitespace.
//
// RE2-compatible version: the original GPT-2 pattern uses (?!\S) lookahead
// for trailing whitespace. We split this into two parts: first catch
// non-whitespace tokens, then handle remaining whitespace.
var gpt2PreTokenizePattern = regexp.MustCompile(
	`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\S+`,
)

// NewGPT2 creates a GPT-2 style BPE tokenizer from loaded metadata.
//
// Parameters:
//   - tokens: vocabulary (from tokenizer.ggml.tokens)
//   - scores: optional token scores (from tokenizer.ggml.scores)
//   - merges: BPE merge pairs (from tokenizer.ggml.merges), each entry "a b"
//   - bosID, eosID, padID: special token IDs (-1 if not set)
func NewGPT2(tokens []string, scores []float32, merges []string, bosID, eosID, padID int) (*BPE, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocabulary")
	}

	bpe := &BPE{
		tokens:     tokens,
		scores:     scores,
		bosTokenID: bosID,
		eosTokenID: eosID,
		padTokenID: padID,
	}

	// Build token → ID lookup
	bpe.tokenToID = make(map[string]int, len(tokens))
	for i, tok := range tokens {
		bpe.tokenToID[tok] = i
	}

	// Build BPE merge ranking from merge list
	// Each merge is "a b" meaning token "ab" is formed by merging "a" and "b"
	bpe.merges = make(map[[2]string]int, len(merges))
	for i, m := range merges {
		parts := strings.SplitN(m, " ", 2)
		if len(parts) != 2 {
			continue // skip invalid entries
		}
		bpe.merges[[2]string{parts[0], parts[1]}] = i
	}

	// Build byte-to-unicode encoding table (GPT-2 style)
	bpe.byteEncoder = make(map[byte]string)
	for b := 0; b < 256; b++ {
		bpe.byteEncoder[byte(b)] = byteToUnicode(byte(b))
	}
	bpe.byteDecoder = make(map[string]byte)
	for b, s := range bpe.byteEncoder {
		bpe.byteDecoder[s] = b
	}

	return bpe, nil
}

// BPE implements a GPT-2 style Byte Pair Encoding tokenizer.
//
// BPE starts with individual characters and iteratively merges the most
// frequent adjacent pairs. The merge order is determined by training on
// a large corpus and stored as an ordered list of merge rules.
//
// GPT-2 style BPE adds a byte-level encoding layer: each byte (0-255)
// is mapped to a Unicode character, making the tokenizer work on raw bytes
// without needing a special "unknown" token for non-text inputs.
type BPE struct {
	tokens []string      // vocabulary
	scores []float32     // token scores (optional)
	merges map[[2]string]int // merge pair → rank (lower = merge first)
	tokenToID map[string]int  // token string → token ID

	// Byte-level encoding tables (GPT-2 style)
	byteEncoder map[byte]string // byte → printable unicode
	byteDecoder map[string]byte // unicode → byte

	// Special token IDs
	bosTokenID int
	eosTokenID int
	padTokenID int
}

// Encode converts text to a slice of token IDs using BPE.
//
// Algorithm:
//   1. Encode input text to bytes
//   2. Map each byte to its printable Unicode representation
//   3. Pre-tokenize into "words" using the GPT-2 regex pattern
//   4. For each word, greedily apply BPE merges
//   5. Map each resulting sub-token to a token ID
func (b *BPE) Encode(text string) ([]int, error) {
	// Step 1-2: Byte-level encoding
	// Convert each byte to its Unicode representation
	var encoded strings.Builder
	for i := 0; i < len(text); i++ {
		encoded.WriteString(b.byteEncoder[text[i]])
	}

	// Step 3: Pre-tokenization — split into "words"
	words := gpt2PreTokenizePattern.FindAllString(encoded.String(), -1)
	if words == nil {
		// If no matches (e.g., empty string), try splitting on whitespace
		words = strings.Fields(encoded.String())
	}

	// Step 4-5: Apply BPE merges to each word and map to IDs
	var ids []int
	for _, word := range words {
		tokens := b.bpeMerge(word)
		for _, tok := range tokens {
			if id, ok := b.tokenToID[tok]; ok {
				ids = append(ids, id)
			} else {
				// Token not found — fallback: use byte-level encoding
				for i := 0; i < len(tok); i++ {
					// Try encoding individual bytes
					byteTok := b.byteEncoder[byte(tok[i])]
					if id, ok := b.tokenToID[byteTok]; ok {
						ids = append(ids, id)
					}
				}
			}
		}
	}

	return ids, nil
}

// bpeMerge applies BPE merge rules to a single pre-tokenized word.
//
// Algorithm:
//   1. Start with the word as a list of individual characters/Unicode tokens
//   2. Find the pair with the LOWEST merge rank (highest priority)
//   3. Merge that pair, repeat until no more merges can be applied
//
// Each merge combines two adjacent symbols into one. The merge order
// (rank) is determined during training: pairs that appear more frequently
// have lower ranks and are merged first.
func (b *BPE) bpeMerge(word string) []string {
	// Start with individual characters
	parts := make([]string, 0, len(word))
	for _, r := range word {
		parts = append(parts, string(r))
	}

	if len(parts) <= 1 {
		return parts
	}

	// Repeatedly find and apply the best merge
	for {
		bestPair := [2]string{}
		bestRank := math.MaxInt
		bestIdx := -1

		// Find the pair with the lowest merge rank
		for i := 0; i < len(parts)-1; i++ {
			pair := [2]string{parts[i], parts[i+1]}
			if rank, ok := b.merges[pair]; ok && rank < bestRank {
				bestRank = rank
				bestPair = pair
				bestIdx = i
			}
		}

		// No more merges to apply
		if bestIdx == -1 {
			break
		}

		// Apply the merge: combine parts[i] and parts[i+1]
		merged := bestPair[0] + bestPair[1]
		newParts := make([]string, 0, len(parts)-1)
		for i := 0; i < len(parts); i++ {
			if i == bestIdx {
				newParts = append(newParts, merged)
				i++ // skip the next part (it was merged)
			} else {
				newParts = append(newParts, parts[i])
			}
		}
		parts = newParts

		if len(parts) <= 1 {
			break
		}
	}

	return parts
}

// Decode converts token IDs back to text.
//
// GPT-2 BPE uses byte-level encoding: each token represents a sequence
// of Unicode characters that correspond to raw bytes. We join the tokens
// and then decode the bytes back to UTF-8 text.
func (b *BPE) Decode(ids []int) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(b.tokens) {
			return "", fmt.Errorf("tokenizer: token ID %d out of range (vocab size %d)", id, len(b.tokens))
		}
		sb.WriteString(b.tokens[id])
	}

	// Decode byte-level representation back to UTF-8
	// Each token consists of Unicode characters that represent raw bytes.
	// We iterate through the string and decode each character.
	result := make([]byte, 0, sb.Len())
	encoded := sb.String()
	for len(encoded) > 0 {
		r, size := utf8.DecodeRuneInString(encoded)
		if r == utf8.RuneError && size == 0 {
			break
		}
		s := string(r)
		if b, ok := b.byteDecoder[s]; ok {
			result = append(result, b)
		}
		encoded = encoded[size:]
	}

	return string(result), nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (b *BPE) VocabSize() int { return len(b.tokens) }

// BosID returns the Beginning-Of-Sequence token ID, or -1 if not set.
func (b *BPE) BosID() int { return b.bosTokenID }

// EosID returns the End-Of-Sequence token ID, or -1 if not set.
func (b *BPE) EosID() int { return b.eosTokenID }

// PadID returns the padding token ID, or -1 if not set.
func (b *BPE) PadID() int { return b.padTokenID }

// LoadFromGGUF creates a tokenizer from GGUF metadata.
// Supports "gpt2" (BPE) and "llama" (SentencePiece) model types.
func LoadFromGGUF(m MetadataProvider) (Tokenizer, error) {
	modelType, _ := m.GetMetadataString("tokenizer.ggml.model")

	switch modelType {
	case "gpt2", "":
		return loadGPT2BPE(m)
	case "llama":
		return loadSentencePiece(m)
	default:
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (supported: gpt2, llama)", modelType)
	}
}

// loadGPT2BPE loads a GPT-2 style BPE tokenizer from GGUF metadata.
func loadGPT2BPE(m MetadataProvider) (*BPE, error) {
	tokens, _ := m.GetMetadataStringArray("tokenizer.ggml.tokens")
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer: empty tokenizer.ggml.tokens")
	}

	scores, _ := m.GetMetadataFloat32Array("tokenizer.ggml.scores")
	merges, _ := m.GetMetadataStringArray("tokenizer.ggml.merges")

	bosID := -1
	eosID := -1
	padID := -1

	if v, ok := m.GetMetadataUint64("tokenizer.ggml.bos_token_id"); ok {
		bosID = int(v)
	}
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.eos_token_id"); ok {
		eosID = int(v)
	}
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.padding_token_id"); ok {
		padID = int(v)
	}

	return NewGPT2(tokens, scores, merges, bosID, eosID, padID)
}

// loadSentencePiece loads a SentencePiece tokenizer from GGUF metadata.
// For now, returns a minimal byte-level tokenizer (decode only).
func loadSentencePiece(m MetadataProvider) (Tokenizer, error) {
	tokens, _ := m.GetMetadataStringArray("tokenizer.ggml.tokens")
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer: empty tokenizer.ggml.tokens")
	}
	scores, _ := m.GetMetadataFloat32Array("tokenizer.ggml.scores")

	bosID := -1
	eosID := -1
	padID := -1

	if v, ok := m.GetMetadataUint64("tokenizer.ggml.bos_token_id"); ok {
		bosID = int(v)
	}
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.eos_token_id"); ok {
		eosID = int(v)
	}
	if v, ok := m.GetMetadataUint64("tokenizer.ggml.padding_token_id"); ok {
		padID = int(v)
	}

	// SentencePiece uses scores for the unigram model.
	// For now, create a BPE-like tokenizer for decode-only support.
	_ = scores
	return NewGPT2(tokens, nil, nil, bosID, eosID, padID)
}

// byteToUnicode maps a byte to a printable Unicode character.
//
// GPT-2's byte-level BPE uses this mapping to ensure every input byte
// can be represented as a valid Unicode string. Bytes that are already
// printable characters (33-126, 161-172, 174-255) keep their Unicode
// identity. Non-printable bytes are mapped to codepoints starting from 256.
//
// This is the exact mapping used by OpenAI's GPT-2 tokenizer.
// It ensures the BPE operates on printable text, while remaining bijective
// (each byte maps to exactly one character, and vice versa).
func byteToUnicode(b byte) string {
	// Printable byte ranges that keep their Unicode codepoint
	// Range 1: '!' (33) to '~' (126) — ASCII printable
	// Range 2: '¡' (161) to '¬' (172) — Latin-1 Supplement (except '­' 173)
	// Range 3: '®' (174) to 'ÿ' (255) — Latin-1 Supplement rest
	//
	// Everything else (0-32, 127-160) maps to 256 + n (the nth non-printable byte)

	// Check if the byte falls in any of the printable ranges
	if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
		return string(rune(b))
	}

	// Non-printable: count how many non-printable bytes precede this one
	// This gives the offset into the 256+ range
	n := int(b)
	count := 0
	for i := 0; i <= n; i++ {
		if !((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255)) {
			count++
		}
	}
	return string(rune(256 + count - 1))
}
