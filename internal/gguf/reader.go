// Package gguf implements a reader for the GGUF binary format.
//
// GGUF (GGML Universal Format) is the standard file format for storing
// quantized LLM (大语言模型) weights used by llama.cpp and its ecosystem.
// It is a binary format designed for fast loading and mmap compatibility.
//
// File structure:
//   ┌──────────────────┐
//   │     Header       │  ← magic, version, tensor/metadata counts
//   │  Metadata KV     │  ← model config (architecture, dimensions, etc.)
//   │  Tensor Info     │  ← name, shape, type, data offset for each tensor
//   │  Padding         │  ← aligned to ALIGNMENT (default 32)
//   │  Tensor Data     │  ← raw weight data (potentially quantized)
//   └──────────────────┘
//
// This package focuses on reading and dequantizing weights into float32
// Tensors that can be used with Minfer's compute.Backend.
//
// For the full specification, see:
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/yusiwen/minfer/internal/model"
)

// GGML type constants.
// These describe the storage format of tensor weights.
//
// For quantized types, weights are stored in "blocks":
// a block contains a scale factor followed by multiple quantized values.
// The block size depends on the quantization scheme.
const (
	TypeF32  GGMLType = 0  // float32 (4 bytes per element)
	TypeF16  GGMLType = 1  // float16 (2 bytes per element)
	TypeQ4_0 GGMLType = 2  // 4-bit block quantization
	TypeQ4_1 GGMLType = 3  // 4-bit block quantization (higher precision)
	TypeQ5_0 GGMLType = 6  // 5-bit block quantization
	TypeQ5_1 GGMLType = 7  // 5-bit block quantization (higher precision)
	TypeQ8_0 GGMLType = 8  // 8-bit block quantization
	TypeQ8_1 GGMLType = 9  // 8-bit block quantization (higher precision)
)

// Metadata value types (from the spec)
const (
	MetaTypeUint8   MetadataValueType = 0
	MetaTypeInt8    MetadataValueType = 1
	MetaTypeUint16  MetadataValueType = 2
	MetaTypeInt16   MetadataValueType = 3
	MetaTypeUint32  MetadataValueType = 4
	MetaTypeInt32   MetadataValueType = 5
	MetaTypeFloat32 MetadataValueType = 6
	MetaTypeBool    MetadataValueType = 7
	MetaTypeString  MetadataValueType = 8
	MetaTypeArray   MetadataValueType = 9
	MetaTypeUint64  MetadataValueType = 10
	MetaTypeInt64   MetadataValueType = 11
	MetaTypeFloat64 MetadataValueType = 12
)

// GGMLType represents the storage type of a tensor's weights.
type GGMLType uint32

// MetadataValueType represents the type of a metadata value.
type MetadataValueType uint32

// BlockSize returns the number of elements per block for quantized types.
// For unquantized types (F32, F16), returns 1 (no blocking).
func (t GGMLType) BlockSize() int {
	switch t {
	case TypeQ4_0, TypeQ4_1:
		return 32
	case TypeQ5_0, TypeQ5_1:
		return 32
	case TypeQ8_0, TypeQ8_1:
		return 32
	default:
		return 1 // F32, F16 — no blocking
	}
}

// TypeSize returns the number of bytes per block for quantized types,
// or bytes per element for unquantized types.
func (t GGMLType) TypeSize() int {
	switch t {
	case TypeF32:
		return 4
	case TypeF16:
		return 2
	case TypeQ4_0:
		return 2 + 16 // f16 scale + 16 bytes of nibbles (32 × 4-bit)
	case TypeQ4_1:
		return 2 + 2 + 16 // f16 scale + f16 min + 16 bytes nibbles
	case TypeQ5_0:
		return 2 + 4 + 16 // f16 scale + 4 bytes (32 high bits) + 16 bytes (32 low nibbles)
	case TypeQ5_1:
		return 2 + 2 + 4 + 16 // f16 scale + f16 min + 4 bytes (32 high bits) + 16 bytes (32 low nibbles)
	case TypeQ8_0:
		return 2 + 32 // f16 scale + 32 × int8
	case TypeQ8_1:
		return 4 + 32 // f32 scale + 32 × int8
	default:
		return 4
	}
}

// String returns the human-readable name of the type.
func (t GGMLType) String() string {
	switch t {
	case TypeF32:
		return "F32"
	case TypeF16:
		return "F16"
	case TypeQ4_0:
		return "Q4_0"
	case TypeQ4_1:
		return "Q4_1"
	case TypeQ5_0:
		return "Q5_0"
	case TypeQ5_1:
		return "Q5_1"
	case TypeQ8_0:
		return "Q8_0"
	case TypeQ8_1:
		return "Q8_1"
	default:
		return fmt.Sprintf("unknown(%d)", t)
	}
}

// TensorInfo describes a single tensor's metadata in the GGUF file.
type TensorInfo struct {
	Name       string   // tensor name (e.g. "blk.0.attn_q.weight")
	Dimensions []uint64 // shape (e.g. [4096, 4096])
	Type       GGMLType // storage type (F32, F16, Q4_0, etc.)
	Offset     uint64   // byte offset relative to the start of tensor_data
}

// Reader reads GGUF files and provides access to metadata and tensor weights.
type Reader struct {
	r io.ReadSeeker

	// Header fields
	Magic       uint32
	Version     uint32
	TensorCount uint64
	MetaKVCount uint64

	// Parsed metadata
	Metadata map[string]any

	// Tensor index
	TensorInfos []TensorInfo

	// Byte offset where tensor_data begins
	DataOffset uint64

	// Alignment (from metadata or default)
	alignment uint64
}

// Open opens a GGUF file for reading.
func Open(r io.ReadSeeker) (*Reader, error) {
	rd := &Reader{
		r:         r,
		Metadata:  make(map[string]any),
		alignment: 32, // default alignment per spec
	}
	if err := rd.readHeader(); err != nil {
		return nil, err
	}
	return rd, nil
}

// readHeader reads and parses the complete GGUF header (magic → tensor infos).
func (r *Reader) readHeader() error {
	// --- Magic + version ---
	if err := binary.Read(r.r, binary.LittleEndian, &r.Magic); err != nil {
		return fmt.Errorf("gguf: reading magic: %w", err)
	}
	if r.Magic != 0x46554747 {
		return fmt.Errorf("gguf: invalid magic: 0x%08X, expected 0x46554747 (GGUF)", r.Magic)
	}

	if err := binary.Read(r.r, binary.LittleEndian, &r.Version); err != nil {
		return fmt.Errorf("gguf: reading version: %w", err)
	}

	// --- Tensor + metadata counts ---
	if err := binary.Read(r.r, binary.LittleEndian, &r.TensorCount); err != nil {
		return fmt.Errorf("gguf: reading tensor count: %w", err)
	}
	if err := binary.Read(r.r, binary.LittleEndian, &r.MetaKVCount); err != nil {
		return fmt.Errorf("gguf: reading metadata count: %w", err)
	}

	// --- Metadata KV pairs ---
	for i := uint64(0); i < r.MetaKVCount; i++ {
		key, err := r.readString()
		if err != nil {
			return fmt.Errorf("gguf: reading metadata key %d: %w", i, err)
		}
		value, err := r.readMetadataValue()
		if err != nil {
			return fmt.Errorf("gguf: reading metadata value for %q: %w", key, err)
		}
		r.Metadata[key] = value

		// Capture alignment from metadata
		if key == "general.alignment" {
			if align, ok := value.(uint32); ok {
				r.alignment = uint64(align)
			}
		}
	}

	// --- Tensor info entries ---
	r.TensorInfos = make([]TensorInfo, r.TensorCount)
	for i := uint64(0); i < r.TensorCount; i++ {
		ti, err := r.readTensorInfo()
		if err != nil {
			return fmt.Errorf("gguf: reading tensor info %d: %w", i, err)
		}
		r.TensorInfos[i] = ti
	}

	// --- Record data offset (after tensor infos + alignment padding) ---
	pos, err := r.r.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("gguf: getting current position: %w", err)
	}
	r.DataOffset = alignOffset(uint64(pos), r.alignment)

	return nil
}

// readString reads a GGUF string (length-prefixed UTF-8).
func (r *Reader) readString() (string, error) {
	var length uint64
	if err := binary.Read(r.r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readMetadataValue reads a single metadata value.
func (r *Reader) readMetadataValue() (any, error) {
	var valueType uint32
	if err := binary.Read(r.r, binary.LittleEndian, &valueType); err != nil {
		return nil, err
	}

	switch MetadataValueType(valueType) {
	case MetaTypeUint8:
		var v uint8
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeInt8:
		var v int8
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeUint16:
		var v uint16
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeInt16:
		var v int16
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeUint32:
		var v uint32
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeInt32:
		var v int32
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeFloat32:
		var v float32
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeBool:
		var v [1]byte
		if _, err := io.ReadFull(r.r, v[:]); err != nil {
			return nil, err
		}
		return v[0] != 0, nil
	case MetaTypeString:
		return r.readString()
	case MetaTypeArray:
		return r.readArray()
	case MetaTypeUint64:
		var v uint64
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeInt64:
		var v int64
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	case MetaTypeFloat64:
		var v float64
		return v, binary.Read(r.r, binary.LittleEndian, &v)
	default:
		return nil, fmt.Errorf("gguf: unknown metadata value type: %d", valueType)
	}
}

// readArray reads a metadata array value.
func (r *Reader) readArray() (any, error) {
	var elemType uint32
	if err := binary.Read(r.r, binary.LittleEndian, &elemType); err != nil {
		return nil, err
	}
	var length uint64
	if err := binary.Read(r.r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}

	// For now, we handle the most common array types.
	// Full generic array handling would require recursive type dispatch.
	switch MetadataValueType(elemType) {
	case MetaTypeString:
		result := make([]string, length)
		for i := uint64(0); i < length; i++ {
			s, err := r.readString()
			if err != nil {
				return nil, err
			}
			result[i] = s
		}
		return result, nil
	case MetaTypeFloat32:
		result := make([]float32, length)
		for i := uint64(0); i < length; i++ {
			if err := binary.Read(r.r, binary.LittleEndian, &result[i]); err != nil {
				return nil, err
			}
		}
		return result, nil
	case MetaTypeUint32:
		result := make([]uint32, length)
		for i := uint64(0); i < length; i++ {
			if err := binary.Read(r.r, binary.LittleEndian, &result[i]); err != nil {
				return nil, err
			}
		}
		return result, nil
	case MetaTypeInt32:
		result := make([]int32, length)
		for i := uint64(0); i < length; i++ {
			if err := binary.Read(r.r, binary.LittleEndian, &result[i]); err != nil {
				return nil, err
			}
		}
		return result, nil
	default:
		return nil, fmt.Errorf("gguf: unsupported array element type: %d", elemType)
	}
}

// readTensorInfo reads a single tensor info entry.
func (r *Reader) readTensorInfo() (TensorInfo, error) {
	name, err := r.readString()
	if err != nil {
		return TensorInfo{}, fmt.Errorf("reading tensor name: %w", err)
	}

	var nDims uint32
	if err := binary.Read(r.r, binary.LittleEndian, &nDims); err != nil {
		return TensorInfo{}, fmt.Errorf("reading tensor dim count for %q: %w", name, err)
	}

	dims := make([]uint64, nDims)
	for i := uint32(0); i < nDims; i++ {
		if err := binary.Read(r.r, binary.LittleEndian, &dims[i]); err != nil {
			return TensorInfo{}, fmt.Errorf("reading tensor dim %d for %q: %w", i, name, err)
		}
	}

	var ggmlType uint32
	if err := binary.Read(r.r, binary.LittleEndian, &ggmlType); err != nil {
		return TensorInfo{}, fmt.Errorf("reading tensor type for %q: %w", name, err)
	}

	var offset uint64
	if err := binary.Read(r.r, binary.LittleEndian, &offset); err != nil {
		return TensorInfo{}, fmt.Errorf("reading tensor offset for %q: %w", name, err)
	}

	return TensorInfo{
		Name:       name,
		Dimensions: dims,
		Type:       GGMLType(ggmlType),
		Offset:     offset,
	}, nil
}

// ReadTensor reads and dequantizes a tensor's weight data by name.
// Returns a float32 Tensor ready for use with Minfer's compute backends.
func (r *Reader) ReadTensor(name string) ([]float32, error) {
	// Find the tensor info
	var info *TensorInfo
	for i := range r.TensorInfos {
		if r.TensorInfos[i].Name == name {
			info = &r.TensorInfos[i]
			break
		}
	}
	if info == nil {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}

	// Calculate number of elements from dimensions
	nElements := uint64(1)
	for _, d := range info.Dimensions {
		nElements *= d
	}

	// Allocate output
	result := make([]float32, nElements)

	// Seek to the tensor's data
	dataPos := int64(r.DataOffset + info.Offset)
	if _, err := r.r.Seek(dataPos, io.SeekStart); err != nil {
		return nil, fmt.Errorf("gguf: seeking to tensor %q data: %w", name, err)
	}

	// Read and dequantize
	switch info.Type {
	case TypeF32:
		if err := binary.Read(r.r, binary.LittleEndian, result); err != nil {
			return nil, fmt.Errorf("gguf: reading F32 tensor %q: %w", name, err)
		}
	case TypeF16:
		if err := r.readF16(result); err != nil {
			return nil, fmt.Errorf("gguf: reading F16 tensor %q: %w", name, err)
		}
	case TypeQ4_0:
		if err := r.readQ4_0(result); err != nil {
			return nil, fmt.Errorf("gguf: reading Q4_0 tensor %q: %w", name, err)
		}
	case TypeQ8_0:
		if err := r.readQ8_0(result); err != nil {
			return nil, fmt.Errorf("gguf: reading Q8_0 tensor %q: %w", name, err)
		}
	default:
		return nil, fmt.Errorf("gguf: unsupported tensor type %s for %q", info.Type, name)
	}

	return result, nil
}

// readF16 reads float16 data and converts to float32 in-place.
func (r *Reader) readF16(dst []float32) error {
	buf := make([]uint16, len(dst))
	if err := binary.Read(r.r, binary.LittleEndian, buf); err != nil {
		return err
	}
	for i, v := range buf {
		dst[i] = f16ToF32(v)
	}
	return nil
}

// readQ4_0 reads Q4_0 quantized data and dequantizes to float32.
//
// Q4_0 block layout (1 block = 32 elements):
//   [0..1]   f16 scale (block-wide scale factor)
//   [2..17]  16 bytes of 4-bit values (2 values per byte, low nibble first)
//
// Dequantization: value[i] = (nibble[i] - 8) * scale
// The -8 shifts the 4-bit unsigned range [0,15] to signed [-8,7].
func (r *Reader) readQ4_0(dst []float32) error {
	n := len(dst)
	blockSize := 32
	buf := make([]byte, 18) // 2 (f16 scale) + 16 (32 nibbles)

	for i := 0; i < n; i += blockSize {
		if _, err := io.ReadFull(r.r, buf); err != nil {
			return err
		}
		scale := f16ToF32(binary.LittleEndian.Uint16(buf[0:2]))
		for j := 0; j < blockSize && i+j < n; j++ {
			// Each byte holds 2 nibbles: low nibble = j%2==0, high nibble = j%2==1
			b := buf[2+j/2]
			var nibble uint8
			if j%2 == 0 {
				nibble = b & 0x0F // low nibble
			} else {
				nibble = b >> 4 // high nibble
			}
			dst[i+j] = float32(int8(nibble)-8) * scale
		}
	}
	return nil
}

// readQ8_0 reads Q8_0 quantized data and dequantizes to float32.
//
// Q8_0 block layout (1 block = 32 elements):
//   [0..1]   f16 scale (block-wide scale factor)
//   [2..33]  32 int8 values
//
// Dequantization: value[i] = int8_value[i] * scale
func (r *Reader) readQ8_0(dst []float32) error {
	n := len(dst)
	blockSize := 32
	buf := make([]byte, 34) // 2 (f16 scale) + 32 (int8 values)

	for i := 0; i < n; i += blockSize {
		if _, err := io.ReadFull(r.r, buf); err != nil {
			return err
		}
		scale := f16ToF32(binary.LittleEndian.Uint16(buf[0:2]))
		for j := 0; j < blockSize && i+j < n; j++ {
			dst[i+j] = float32(int8(buf[2+j])) * scale
		}
	}
	return nil
}

// Architecture returns the model architecture string (e.g. "llama", "qwen2").
func (r *Reader) Architecture() string {
	if v, ok := r.Metadata["general.architecture"]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// GetMetadataString retrieves a metadata value as a string.
func (r *Reader) GetMetadataString(key string) (string, bool) {
	v, ok := r.Metadata[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// GetMetadataUint64 retrieves a metadata value as uint64.
// Supports uint32, uint64, int32, and int64 metadata types.
func (r *Reader) GetMetadataUint64(key string) (uint64, bool) {
	v, ok := r.Metadata[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case uint64:
		return val, true
	case uint32:
		return uint64(val), true
	case int64:
		return uint64(val), true
	case int32:
		return uint64(val), true
	default:
		return 0, false
	}
}

// GetMetadataFloat32 retrieves a metadata value as float32.
func (r *Reader) GetMetadataFloat32(key string) (float32, bool) {
	v, ok := r.Metadata[key]
	if !ok {
		return 0, false
	}
	f, ok := v.(float32)
	return f, ok
}

// GetMetadataStringArray retrieves a metadata value as a []string.
func (r *Reader) GetMetadataStringArray(key string) ([]string, bool) {
	v, ok := r.Metadata[key]
	if !ok {
		return nil, false
	}
	s, ok := v.([]string)
	return s, ok
}

// GetMetadataFloat32Array retrieves a metadata value as a []float32.
func (r *Reader) GetMetadataFloat32Array(key string) ([]float32, bool) {
	v, ok := r.Metadata[key]
	if !ok {
		return nil, false
	}
	s, ok := v.([]float32)
	return s, ok
}

// --- Config loading ---

// LoadConfig reads model architecture parameters from GGUF metadata
// and returns a model.Config populated with the discovered values.
//
// The architecture prefix (e.g. "llama", "qwen2") is determined from
// the "general.architecture" metadata key.
func (r *Reader) LoadConfig() (model.Config, error) {
	arch := r.Architecture()
	if arch == "" {
		return model.Config{}, fmt.Errorf("gguf: missing general.architecture metadata")
	}

	cfg := model.Config{
		RoPEBase:   10000.0,
		NormEpsilon: 1e-6,
	}

	// Helper: read uint64 metadata with architecture prefix, with fallback
	readUint := func(key string) (uint64, bool) {
		if v, ok := r.GetMetadataUint64(arch + "." + key); ok {
			return v, true
		}
		// Fallback: try with "llama." prefix for cross-architecture models
		if arch != "llama" {
			return r.GetMetadataUint64("llama." + key)
		}
		return 0, false
	}

	readFloat := func(key string) (float32, bool) {
		if v, ok := r.GetMetadataFloat32(arch + "." + key); ok {
			return v, true
		}
		if arch != "llama" {
			return r.GetMetadataFloat32("llama." + key)
		}
		return 0, false
	}

	if v, ok := readUint("block_count"); ok {
		cfg.NumLayers = int(v)
	}
	if v, ok := readUint("embedding_length"); ok {
		cfg.HiddenDim = int(v)
	}
	if v, ok := readUint("feed_forward_length"); ok {
		cfg.FFNHiddenDim = int(v)
	}
	if v, ok := readUint("context_length"); ok {
		cfg.MaxSeqLen = int(v)
	}
	if v, ok := readUint("attention.head_count"); ok {
		cfg.NumHeads = int(v)
	}
	if v, ok := readUint("attention.head_count_kv"); ok {
		cfg.NumKVHeads = int(v)
	} else {
		cfg.NumKVHeads = cfg.NumHeads // default: MHA (no GQA)
	}
	if v, ok := readFloat("attention.layer_norm_rms_epsilon"); ok {
		cfg.NormEpsilon = v
	}
	if v, ok := readFloat("rope.freq_base"); ok {
		cfg.RoPEBase = v
	}

	// VocabSize: read from tokenizer metadata (string array length)
	if tokens, ok := r.GetMetadataStringArray("tokenizer.ggml.tokens"); ok {
		cfg.VocabSize = len(tokens)
	}
	// Fallback: try legacy uint64 format
	if cfg.VocabSize == 0 {
		if v, ok := r.GetMetadataUint64("tokenizer.ggml.tokens"); ok {
			cfg.VocabSize = int(v)
		}
	}

	// HeadDim = HiddenDim / NumHeads
	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.HiddenDim / cfg.NumHeads
	}

	return cfg, nil
}

// --- helpers ---

// alignOffset rounds offset up to the nearest multiple of alignment.
func alignOffset(offset, alignment uint64) uint64 {
	return (offset + alignment - 1) / alignment * alignment
}

// f16ToF32 converts a uint16 in IEEE 754 half-precision format to float32.
//
// float16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
// float32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
//
// Subnormal handling uses modular arithmetic:
// The loop decrements exp (uint32) for each normalization shift.
// Starting from exp=0, after k shifts exp wraps to 2^32 - k,
// and (2^32 - k + 113) wraps back to (113 - k), which equals
// the correct f32 exponent (-14 - k + 127) = (113 - k).
// This is well-defined in Go: unsigned integer overflow wraps around.
func f16ToF32(v uint16) float32 {
	// Extract components
	sign := uint32(v>>15) & 1
	exp := uint32(v>>10) & 0x1F
	mant := uint32(v) & 0x03FF

	var f32 uint32

	switch {
	case exp == 0 && mant == 0:
		// Zero
		f32 = sign << 31
	case exp == 0:
		// Subnormal: exp = -14, implicit leading 0
		// Convert to normal float32: adjust exponent and mantissa
		for mant&0x0400 == 0 { // 0x0400 = 0b10000000000 (bit 10)
			mant <<= 1
			exp--
		}
		mant &= 0x03FF // clear the leading 1
		f32 = (sign << 31) | ((exp + 127 - 15 + 1) << 23) | (mant << 13)
	case exp == 31:
		// Infinity or NaN
		f32 = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		// Normal: exp bias is 15 for f16, 127 for f32
		f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}

	return math.Float32frombits(f32)
}
