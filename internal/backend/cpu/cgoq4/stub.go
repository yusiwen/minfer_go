//go:build !(amd64 && !cgo_no_avx2)

package cgoq4

import "unsafe"

// MatmulQ4Row is a no-op stub when CGo+AVX2 is unavailable.
func MatmulQ4Row(a, q4data, c unsafe.Pointer, K, N, colStart, colEnd, blkPerRow int) bool {
	return false
}
