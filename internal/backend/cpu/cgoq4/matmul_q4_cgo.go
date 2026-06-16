//go:build amd64 && !cgo_no_avx2

package cgoq4

/*
#cgo CFLAGS: -O3

#include <stdint.h>

// Declare function for CGo type inference (no body, no AVX2).
void matmul_q4_row_fma(const float* a, const uint8_t* q4_data, float* c,
    int K, int N, int col_start, int col_end, int blk_per_row, int quant_type);
*/
import "C"

import "unsafe"

// MatmulQ4Row calls the fused Q4+AVX2 kernel via CGo.
// quantType: 2 = Q4_0, 8 = Q8_0
func MatmulQ4Row(a, q4data, c unsafe.Pointer, K, N, colStart, colEnd, blkPerRow, quantType int) bool {
	C.matmul_q4_row_fma(
		(*C.float)(a),
		(*C.uint8_t)(q4data),
		(*C.float)(c),
		C.int(K), C.int(N),
		C.int(colStart), C.int(colEnd),
		C.int(blkPerRow), C.int(quantType),
	)
	return true
}
