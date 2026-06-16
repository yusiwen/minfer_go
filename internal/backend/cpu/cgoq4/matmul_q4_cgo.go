//go:build amd64 && !cgo_no_avx2

package cgoq4

/*
#cgo CFLAGS: -O3

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

// f16_to_f32 converts IEEE 754 half-precision to single-precision.
static inline float f16_to_f32(uint16_t v) {
    uint32_t sign = (uint32_t)(v >> 15) & 1;
    uint32_t exp  = (uint32_t)(v >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(v & 0x3FF);
    uint32_t bits;
    if (exp == 0) {
        bits = (sign << 31) | (mant << 13);
        float r; memcpy(&r, &bits, 4);
        return r * (1.0f / (1 << 12));
    }
    if (exp == 31) {
        bits = (sign << 31) | 0x7F800000 | (mant << 13);
        float r; memcpy(&r, &bits, 4);
        return r;
    }
    bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float r; memcpy(&r, &bits, 4);
    return r;
}

// Fused Q4_0 dequant + AVX2 MatMul for one row of A.
// Processed as K rows, each with B packed in Q4_0 blocks.
// Accumulates into C[col_start:col_end].
__attribute__((target("avx2,fma")))
void matmul_q4_row_fma(
    const float* a, const uint8_t* q4_data, float* c,
    int K, int N, int col_start, int col_end, int blk_per_row
) {
    for (int k = 0; k < K; k++) {
        __m256 a_bc = _mm256_set1_ps(a[k]);
        int j = col_start;
        while (j < col_end) {
            int chunk_end = j + 32;
            if (chunk_end > col_end) chunk_end = col_end;
            int chunk_len = chunk_end - j;
            if (chunk_len <= 0) break;

            const uint8_t* blk = q4_data + (k * blk_per_row + j / 32) * 18;
            float scale = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            __m256 scale_v = _mm256_set1_ps(scale);
            __m256 eight_v = _mm256_set1_ps(8.0f);

            int groups = (chunk_len + 7) / 8;
            for (int g = 0; g < groups; g++) {
                int nib_off = 2 + g * 4;
                int nv = 8;
                if (g == groups - 1 && chunk_len % 8 != 0)
                    nv = chunk_len % 8;

                uint32_t nib_word;
                memcpy(&nib_word, blk + nib_off, 4);
                uint32_t lo = nib_word & 0x0F0F0F0F;
                uint32_t hi = (nib_word >> 4) & 0x0F0F0F0F;

                __m128i lo128 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)lo));
                __m128i hi128 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)hi));
                __m256i nib256 = _mm256_unpacklo_epi32(
                    _mm256_castsi128_si256(lo128),
                    _mm256_castsi128_si256(hi128));

                __m256 bf = _mm256_cvtepi32_ps(nib256);
                bf = _mm256_sub_ps(bf, eight_v);
                bf = _mm256_mul_ps(bf, scale_v);

                int c_off = j + g * 8;
                if (nv == 8) {
                    __m256 cv = _mm256_loadu_ps(c + c_off);
                    cv = _mm256_fmadd_ps(a_bc, bf, cv);
                    _mm256_storeu_ps(c + c_off, cv);
                } else {
                    float* bf_s = (float*)&bf;
                    for (int v = 0; v < nv; v++)
                        c[c_off + v] += a[k] * bf_s[v];
                }
            }
            j = chunk_end;
        }
    }
}
*/
import "C"

import "unsafe"

// MatmulQ4Row calls the fused Q4+AVX2 kernel via CGo.
// Processes one row of A against Q4_0 quantized B,
// accumulating into C[colStart:colEnd].
func MatmulQ4Row(a, q4data, c unsafe.Pointer, K, N, colStart, colEnd, blkPerRow int) bool {
	C.matmul_q4_row_fma(
		(*C.float)(a),
		(*C.uint8_t)(q4data),
		(*C.float)(c),
		C.int(K), C.int(N),
		C.int(colStart), C.int(colEnd),
		C.int(blkPerRow),
	)
	return true
}
