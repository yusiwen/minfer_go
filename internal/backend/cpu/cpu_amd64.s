// func matmulAVX2Kernel(a, b, c unsafe.Pointer, K, N, colStart, colEnd int)
//
// Core loop for one row of A:
//   C[colStart:colEnd] += Σ A[k] × B[k*N+colStart : k*N+colEnd]
//
// Uses AVX2 + FMA to process 8 float32 per iteration.
// Steps by 8 in the N dimension; trailing elements (<8) use scalar fallback.

TEXT ·matmulAVX2Kernel(SB), $8-56
	// Save callee-saved register
	MOVQ	BP, (SP)

	// Load arguments
	MOVQ	a+0(FP), AX		// AX = a (pointer to A.data)
	MOVQ	b+8(FP), BX		// BX = b (pointer to B.data)
	MOVQ	c+16(FP), CX		// CX = c (pointer to C.data)
	MOVQ	K+24(FP), DI		// DI = K (inner dimension)
	MOVQ	N+32(FP), SI		// SI = N (number of columns in B)
	MOVQ	colStart+40(FP), R8	// R8 = colStart
	MOVQ	colEnd+48(FP), R9	// R9 = colEnd

	XORQ	R10, R10		// k = 0

loop_k:
	CMPQ	R10, DI			// k >= K?
	JGE	done

	// Load aVal = A[k] and store to temporary slot for VBROADCASTSS
	MOVSS	(AX)(R10*4), X0		// X0 = A[k] (lower 32 bits)
	MOVSS	X0, 4(SP)		// store to temp slot
	VBROADCASTSS 4(SP), Y0		// Y0 = [aVal, aVal, ..., aVal] (8 copies)

	// j = colStart
	MOVQ	R8, R11

loop_j:
	// Check if we can process 8 elements
	LEAQ	8(R11), R12		// R12 = j + 8
	CMPQ	R12, R9			// j + 8 <= colEnd?
	JG	cleanup_scalar

	// Process 8 elements with AVX2:
	//   C[j:j+8] += aVal * B[k*N+j : k*N+j+8]

	// Compute B byte offset: (k*N + j) * 4
	MOVQ	R10, R12		// R12 = k
	IMULQ	SI, R12			// R12 = k * N
	ADDQ	R11, R12		// R12 = k*N + j
	SHLQ	$2, R12			// R12 = (k*N + j) * 4

	// Load 8 floats from B
	VMOVUPS	(BX)(R12*1), Y1		// Y1 = B[k*N+j : k*N+j+8]

	// Compute C byte offset: j * 4
	MOVQ	R11, R13		// R13 = j
	SHLQ	$2, R13			// R13 = j * 4

	// Load 8 floats from C
	VMOVUPS	(CX)(R13*1), Y2		// Y2 = C[j:j+8]

	// FMA: Y2 = Y0 * Y1 + Y2
	VFMADD231PS Y1, Y0, Y2		// Y2 = Y0 * Y1 + Y2

	// Store back to C
	VMOVUPS	Y2, (CX)(R13*1)

	// j += 8
	ADDQ	$8, R11
	JMP	loop_j

cleanup_scalar:
	// Handle remaining elements one at a time
	CMPQ	R11, R9			// j >= colEnd?
	JGE	next_k

scalar_loop:
	// C[j] += aVal * B[k*N + j]
	MOVQ	R10, R12		// R12 = k
	IMULQ	SI, R12			// R12 = k * N
	ADDQ	R11, R12		// R12 = k*N + j
	SHLQ	$2, R12			// R12 = (k*N + j) * 4

	MOVSS	(BX)(R12*1), X1		// X1 = B[k][j]
	MULSS	X0, X1			// X1 = aVal * B[k][j]

	MOVQ	R11, R13		// R13 = j
	SHLQ	$2, R13			// R13 = j * 4

	MOVSS	(CX)(R13*1), X2		// X2 = C[j]
	ADDSS	X1, X2			// X2 += aVal * B[k][j]
	MOVSS	X2, (CX)(R13*1)		// C[j] = X2

	ADDQ	$1, R11
	CMPQ	R11, R9
	JL	scalar_loop

next_k:
	ADDQ	$1, R10			// k++
	JMP	loop_k

done:
	// Restore callee-saved register
	MOVQ	(SP), BP
	RET
