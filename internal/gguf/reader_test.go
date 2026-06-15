package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// buildMinimalGGUF creates a minimal valid GGUF file in memory.
//
// Structure:
//   header: magic (4) + version (4) + tensor_count (8) + metadata_count (8)
//   metadata: 1 KV pair: general.architecture = "test"
//   tensor info: 1 tensor "test.weight", shape [4], type F32
//   padding to ALIGNMENT (32)
//   tensor data: [1.0, 2.0, 3.0, 4.0] as float32
func buildMinimalGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer

	// Header
	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747)) // magic "GGUF"
	binary.Write(&buf, binary.LittleEndian, uint32(3))          // version 3
	binary.Write(&buf, binary.LittleEndian, uint64(1))          // tensor_count = 1
	binary.Write(&buf, binary.LittleEndian, uint64(1))          // metadata_kv_count = 1

	// Metadata: general.architecture = "test"
	writeString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8)) // MetaTypeString
	writeString(&buf, "test")

	// Tensor info: "test.weight", shape [4], type F32
	writeString(&buf, "test.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))   // n_dimensions = 1
	binary.Write(&buf, binary.LittleEndian, uint64(4))   // dims[0] = 4
	binary.Write(&buf, binary.LittleEndian, uint32(0))   // type = F32
	binary.Write(&buf, binary.LittleEndian, uint64(0))   // offset = 0 (first tensor)

	// Padding to alignment (32)
	pos := buf.Len()
	padding := (32 - pos%32) % 32
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}

	// Tensor data: [1.0, 2.0, 3.0, 4.0]
	data := []float32{1.0, 2.0, 3.0, 4.0}
	binary.Write(&buf, binary.LittleEndian, data)

	return buf.Bytes()
}

// buildGGUFF16 creates a GGUF file with an F16 tensor.
func buildGGUFF16(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer

	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 tensor
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 metadata

	writeString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8))
	writeString(&buf, "test")

	// Tensor info: shape [4], type F16
	writeString(&buf, "f16.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(4))
	binary.Write(&buf, binary.LittleEndian, uint32(1)) // type = F16
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	// Padding
	pos := buf.Len()
	padding := (32 - pos%32) % 32
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}

	// F16 data: [1.0, 2.0, 3.0, 4.0]
	vals := []uint16{f32ToF16(1.0), f32ToF16(2.0), f32ToF16(3.0), f32ToF16(4.0)}
	binary.Write(&buf, binary.LittleEndian, vals)

	return buf.Bytes()
}

// buildGGUFQ4_0 creates a GGUF file with a Q4_0 quantized tensor.
func buildGGUFQ4_0(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer

	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint64(1))

	writeString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8))
	writeString(&buf, "test")

	// Tensor: shape [32], type Q4_0 (1 full block)
	writeString(&buf, "q4_0.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(32)) // 32 elements = 1 block
	binary.Write(&buf, binary.LittleEndian, uint32(2))  // type = Q4_0
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	// Padding
	pos := buf.Len()
	padding := (32 - pos%32) % 32
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}

	// Q4_0 block: scale = 1.0, values cycle through [0..15] twice (4-bit nibbles, max 15)
	// Each byte: low nibble = (j*2)%16, high nibble = (j*2+1)%16
	// Nibble sequence: [0,1,2,...,15,0,1,2,...,15]
	scale := f32ToF16(1.0)
	binary.Write(&buf, binary.LittleEndian, uint16(scale))
	for j := 0; j < 16; j++ {
		low := byte(j * 2 % 16)
		high := byte((j*2 + 1) % 16)
		buf.WriteByte(low | (high << 4))
	}

	return buf.Bytes()
}

// TestMagic verifies the GGUF magic number detection.
func TestMagic(t *testing.T) {
	data := buildMinimalGGUF(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if r.Magic != 0x46554747 {
		t.Errorf("magic = 0x%08X, want 0x46554747", r.Magic)
	}
	if r.Version != 3 {
		t.Errorf("version = %d, want 3", r.Version)
	}
}

// TestInvalidMagic verifies that invalid files are rejected.
func TestInvalidMagic(t *testing.T) {
	data := []byte{0, 0, 0, 0, 1, 0, 0, 0} // all zeros
	_, err := Open(bytes.NewReader(data))
	if err == nil {
		t.Fatal("expected error for invalid magic, got nil")
	}
}

// TestMetadata verifies metadata parsing.
func TestMetadata(t *testing.T) {
	data := buildMinimalGGUF(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if r.Architecture() != "test" {
		t.Errorf("architecture = %q, want %q", r.Architecture(), "test")
	}
}

// TestTensorInfos verifies tensor info parsing.
func TestTensorInfos(t *testing.T) {
	data := buildMinimalGGUF(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.TensorInfos) != 1 {
		t.Fatalf("got %d tensor infos, want 1", len(r.TensorInfos))
	}
	ti := r.TensorInfos[0]
	if ti.Name != "test.weight" {
		t.Errorf("tensor name = %q, want %q", ti.Name, "test.weight")
	}
	if len(ti.Dimensions) != 1 || ti.Dimensions[0] != 4 {
		t.Errorf("tensor shape = %v, want [4]", ti.Dimensions)
	}
	if ti.Type != TypeF32 {
		t.Errorf("tensor type = %s, want F32", ti.Type)
	}
}

// TestReadF32 verifies reading an F32 tensor.
func TestReadF32(t *testing.T) {
	data := buildMinimalGGUF(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}

	vals, err := r.ReadTensor("test.weight")
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i, v := range expected {
		if vals[i] != v {
			t.Errorf("val[%d] = %f, want %f", i, vals[i], v)
		}
	}
}

// TestReadF16 verifies reading an F16 tensor.
func TestReadF16(t *testing.T) {
	data := buildGGUFF16(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}

	vals, err := r.ReadTensor("f16.weight")
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i, v := range expected {
		if abs(vals[i]-v) > 1e-3 {
			t.Errorf("val[%d] = %f, want %f (F16 round-trip)", i, vals[i], v)
		}
	}
}

// TestReadQ4_0 verifies reading a Q4_0 quantized tensor.
func TestReadQ4_0(t *testing.T) {
	data := buildGGUFQ4_0(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}

	vals, err := r.ReadTensor("q4_0.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(vals) != 32 {
		t.Fatalf("got %d values, want 32", len(vals))
	}

	// Our block has nibbles cycling [0,1,2,...,15,0,1,2,...,15]
	// With centered range (-8): values = [(0-8)*1.0, (1-8)*1.0, ...] = [-8, -7, ..., 7, -8, -7, ..., 7]
	for i := 0; i < 32; i++ {
		expected := float32(int8(i%16) - 8)
		if vals[i] != expected {
			t.Errorf("val[%d] = %f, want %f", i, vals[i], expected)
		}
	}
}

// TestTensorNotFound verifies error when reading a nonexistent tensor.
func TestTensorNotFound(t *testing.T) {
	data := buildMinimalGGUF(t)
	r, err := Open(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}

	_, err = r.ReadTensor("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent tensor")
	}
}

// TestF16ToF32Conversion verifies the f16 → f32 conversion.
func TestF16ToF32Conversion(t *testing.T) {
	tests := []struct {
		f16  uint16
		want float32
	}{
		{0x0000, 0.0},           // zero
		{0x8000, -0.0},          // negative zero
		{0x3C00, 1.0},           // 1.0
		{0xBC00, -1.0},          // -1.0
		{0x3555, 0.33325195},    // ~1/3 (a lossy approximation in f16)
		{0x7C00, float32(math.Inf(1))},  // +inf
		{0xFC00, float32(math.Inf(-1))}, // -inf
	}

	for _, tt := range tests {
		got := f16ToF32(tt.f16)
		if math.IsInf(float64(tt.want), 1) && math.IsInf(float64(got), 1) {
			continue // both +inf
		}
		if math.IsInf(float64(tt.want), -1) && math.IsInf(float64(got), -1) {
			continue // both -inf
		}
		if abs(got-tt.want) > 1e-3 {
			t.Errorf("f16ToF32(0x%04X) = %f, want %f", tt.f16, got, tt.want)
		}
	}
}

// TestAlignOffset verifies alignment calculation.
func TestAlignOffset(t *testing.T) {
	tests := []struct {
		offset, align, want uint64
	}{
		{0, 32, 0},
		{1, 32, 32},
		{31, 32, 32},
		{32, 32, 32},
		{33, 32, 64},
		{0, 8, 0},
		{7, 8, 8},
	}
	for _, tt := range tests {
		got := alignOffset(tt.offset, tt.align)
		if got != tt.want {
			t.Errorf("alignOffset(%d, %d) = %d, want %d", tt.offset, tt.align, got, tt.want)
		}
	}
}

// TestTypeProperties verifies GGML type sizes and block sizes.
func TestTypeProperties(t *testing.T) {
	tests := []struct {
		typ       GGMLType
		typeSize  int
		blockSize int
	}{
		{TypeF32, 4, 1},
		{TypeF16, 2, 1},
		{TypeQ4_0, 18, 32},
		{TypeQ8_0, 34, 32},
	}

	for _, tt := range tests {
		if ts := tt.typ.TypeSize(); ts != tt.typeSize {
			t.Errorf("%s TypeSize = %d, want %d", tt.typ, ts, tt.typeSize)
		}
		if bs := tt.typ.BlockSize(); bs != tt.blockSize {
			t.Errorf("%s BlockSize = %d, want %d", tt.typ, bs, tt.blockSize)
		}
	}
}

// TestLoadConfig verifies config loading from architecture-specific metadata.
func TestLoadConfig(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint64(8)) // 8 metadata entries

	writeString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8))
	writeString(&buf, "qwen2")

	writeString(&buf, "qwen2.block_count")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(24))

	writeString(&buf, "qwen2.embedding_length")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(896))

	writeString(&buf, "qwen2.feed_forward_length")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(4864))

	writeString(&buf, "qwen2.attention.head_count")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(14))

	writeString(&buf, "qwen2.attention.head_count_kv")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(2))

	writeString(&buf, "qwen2.attention.layer_norm_rms_epsilon")
	binary.Write(&buf, binary.LittleEndian, uint32(6)) // float32
	binary.Write(&buf, binary.LittleEndian, float32(1e-6))

	writeString(&buf, "tokenizer.ggml.tokens")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(151936))

	// Tensor info (dummy)
	writeString(&buf, "dummy.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint32(0))
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	pos := buf.Len()
	padding := (32 - pos%32) % 32
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}
	binary.Write(&buf, binary.LittleEndian, float32(1.0))

	r, err := Open(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := r.LoadConfig()
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HiddenDim != 896 {
		t.Errorf("HiddenDim = %d, want 896", cfg.HiddenDim)
	}
	if cfg.NumLayers != 24 {
		t.Errorf("NumLayers = %d, want 24", cfg.NumLayers)
	}
	if cfg.NumHeads != 14 {
		t.Errorf("NumHeads = %d, want 14", cfg.NumHeads)
	}
	if cfg.NumKVHeads != 2 {
		t.Errorf("NumKVHeads = %d, want 2", cfg.NumKVHeads)
	}
	if cfg.HeadDim != 64 {
		t.Errorf("HeadDim = %d, want 64 (896/14)", cfg.HeadDim)
	}
	if cfg.FFNHiddenDim != 4864 {
		t.Errorf("FFNHiddenDim = %d, want 4864", cfg.FFNHiddenDim)
	}
	if cfg.VocabSize != 151936 {
		t.Errorf("VocabSize = %d, want 151936", cfg.VocabSize)
	}
}

// TestLoadConfigFallback verifies that architecture-specific keys fall back to "llama." prefix.
func TestLoadConfigFallback(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint64(3)) // 3 metadata entries

	// Use unknown architecture with llama-prefixed keys
	writeString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8))
	writeString(&buf, "myarch")

	writeString(&buf, "llama.block_count")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(32))

	writeString(&buf, "llama.embedding_length")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(4096))

	// Tensor info (dummy)
	writeString(&buf, "dummy.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint32(0))
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	pos := buf.Len()
	padding := (32 - pos%32) % 32
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}
	binary.Write(&buf, binary.LittleEndian, float32(1.0))

	r, err := Open(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := r.LoadConfig()
	if err != nil {
		t.Fatal(err)
	}

	if cfg.HiddenDim != 4096 {
		t.Errorf("HiddenDim = %d, want 4096", cfg.HiddenDim)
	}
	if cfg.NumLayers != 32 {
		t.Errorf("NumLayers = %d, want 32", cfg.NumLayers)
	}
}

// TestArchitectureKey verifies reading architecture-specific metadata.
func TestArchitectureKey(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint64(3))

	// 3 metadata keys
	writeString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8))
	writeString(&buf, "qwen2")

	writeString(&buf, "qwen2.block_count")
	binary.Write(&buf, binary.LittleEndian, uint32(4)) // uint32
	binary.Write(&buf, binary.LittleEndian, uint32(24))

	writeString(&buf, "qwen2.embedding_length")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(896))

	// Tensor info (dummy)
	writeString(&buf, "dummy.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	binary.Write(&buf, binary.LittleEndian, uint32(0)) // F32
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	// Padding + data
	pos := buf.Len()
	padding := (32 - pos%32) % 32
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}
	binary.Write(&buf, binary.LittleEndian, float32(42.0))

	r, err := Open(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	if r.Architecture() != "qwen2" {
		t.Errorf("architecture = %q, want %q", r.Architecture(), "qwen2")
	}

	nLayers, ok := r.GetMetadataUint64("qwen2.block_count")
	if !ok || nLayers != 24 {
		t.Errorf("block_count = %d, want 24", nLayers)
	}

	hiddenDim, ok := r.GetMetadataUint64("qwen2.embedding_length")
	if !ok || hiddenDim != 896 {
		t.Errorf("embedding_length = %d, want 896", hiddenDim)
	}
}

// --- helpers ---

func writeString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

func f32ToF16(v float32) uint16 {
	f := math.Float32bits(v)
	sign := uint16((f >> 31) & 1)
	exp := int32((f >> 23) & 0xFF)
	mant := f & 0x007FFFFF

	var out uint16

	switch {
	case exp == 0 && mant == 0:
		// Zero
		out = sign << 15
	case exp == 0xFF:
		// NaN or Inf
		out = (sign << 15) | (0x1F << 10) | uint16(mant>>13)
	case exp < 113: // exp - 127 < -14 → subnormal in f16
		// Too small to represent as normal f16 → flush to zero
		out = sign << 15
	case exp > 142: // exp - 127 > 15 → overflow to inf
		out = (sign << 15) | (0x1F << 10)
	default:
		out = (sign << 15) | uint16((exp-127+15)<<10) | uint16(mant>>13)
	}

	return out
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
