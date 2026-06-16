// AVX2 MatMul kernel for float32 data.
//
// Processes one row of A against float32 B, for column range [colStart, colEnd).
// Uses VFMADD231PS to do 8-wide FMA.
//
// func matmulAVX2Kernel(a, b, c unsafe.Pointer, K, N, colStart, colEnd int)

TEXT ·matmulAVX2Kernel(SB), $8-56
	// Load arguments
	MOVQ	a+0(FP), AX		// AX = a
	MOVQ	b+8(FP), BX		// BX = b
	MOVQ	c+16(FP), CX		// CX = c
	MOVQ	K+24(FP), DI		// DI = K
	MOVQ	N+32(FP), SI		// SI = N
	MOVQ	colStart+40(FP), R8	// R8 = colStart
	MOVQ	colEnd+48(FP), R9	// R9 = colEnd

	XORQ	R10, R10		// k = 0

loop_k:
	CMPQ	R10, DI
	JGE	done

	// Load aVal = A[k] and broadcast to Y0
	MOVSS	(AX)(R10*4), X0
	MOVSS	X0, (SP)
	VBROADCASTSS (SP), Y0

	// j = colStart
	MOVQ	R8, R11

loop_j:
	LEAQ	8(R11), R12
	CMPQ	R12, R9
	JG	cleanup_scalar

	// Compute B offset: (k*N + j) * 4
	MOVQ	R10, R12
	IMULQ	SI, R12
	ADDQ	R11, R12
	SHLQ	$2, R12

	// Load 8 B values
	VMOVUPS	(BX)(R12*1), Y1

	// Compute C offset: j * 4
	MOVQ	R11, R13
	SHLQ	$2, R13

	// Load 8 C values
	VMOVUPS	(CX)(R13*1), Y2

	// FMA: Y2 = Y0 * Y1 + Y2
	VFMADD231PS Y1, Y0, Y2

	// Store back to C
	VMOVUPS	Y2, (CX)(R13*1)

	ADDQ	$8, R11
	JMP	loop_j

cleanup_scalar:
	CMPQ	R11, R9
	JGE	next_k

scalar_loop:
	MOVQ	R10, R12
	IMULQ	SI, R12
	ADDQ	R11, R12
	SHLQ	$2, R12

	MOVSS	(BX)(R12*1), X1
	MULSS	X0, X1

	MOVQ	R11, R13
	SHLQ	$2, R13

	MOVSS	(CX)(R13*1), X2
	ADDSS	X1, X2
	MOVSS	X2, (CX)(R13*1)

	ADDQ	$1, R11
	CMPQ	R11, R9
	JL	scalar_loop

next_k:
	ADDQ	$1, R10
	JMP	loop_k

done:
	RET
