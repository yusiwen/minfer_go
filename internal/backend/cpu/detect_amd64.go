// +build amd64

package cpu

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// HasAVX2 reports whether the current CPU supports AVX2+FMA instructions.
func HasAVX2() bool { return cpu.X86.HasAVX2 && cpu.X86.HasFMA }

//go:noescape
//go:nosplit
func matmulAVX2Kernel(a, b, c unsafe.Pointer, K, N, colStart, colEnd int)

// TryMatmulAVX2 attempts AVX2-accelerated MatMul for one worker's column range.
// B is float32 data. Returns true if the kernel was used, false for Go fallback.
func TryMatmulAVX2(a, b, c unsafe.Pointer, K, N, colStart, colEnd int) bool {
	if !HasAVX2() {
		return false
	}
	matmulAVX2Kernel(a, b, c, K, N, colStart, colEnd)
	return true
}
