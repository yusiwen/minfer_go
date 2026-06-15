# Minfer — Go LLM Inference Engine from Scratch

Minfer is a from-scratch LLM local inference engine written in Go.  
The primary goal is **educational** — to deeply understand every layer of LLM inference by implementing it manually.

## Philosophy

- **No mature frameworks** — we write our own tensor operations, GGUF parser, tokenizer, transformer, and sampler
- **Backend-agnostic architecture** — the `ComputeBackend` interface makes it easy to add GPU support later
- **Learn by building** — every line of code has detailed Chinese comments explaining the math

## Status

**Phase 1: Tensor Layer — ✅ Complete**

- `Tensor` data type (row-major storage, any dimension)
- Shape operations (`View`, `Clone`, `At`, `Set`)
- `ComputeBackend` interface (`MatMul`, `Softmax`, `RMSNorm`, `RoPE`, `SiLU`, `Add`)
- `CPUBackend` implementation (pure Go, no dependencies)

## Quick Start

```bash
make build
./bin/minfer version
./bin/minfer run --backend cpu model.gguf
```

## Project Map

```
minfer/
├── cmd/minfer/           — CLI entry point
├── internal/
│   ├── tensor/           — Tensor data type
│   ├── compute/          — ComputeBackend interface
│   ├── backend/cpu/      — Pure Go CPU backend
│   ├── gguf/             — GGUF parser (WIP)
│   ├── tokenizer/        — BPE / SentencePiece (WIP)
│   ├── model/            — Transformer model (WIP)
│   ├── infer/            — Inference engine (WIP)
│   └── registry/         — Model download (WIP)
├── Makefile
└── README.md
```

## Hardware Plan

| Device | Backend | Status |
|--------|---------|--------|
| Any CPU | CPUBackend (pure Go) | ✅ Phase 1 done |
| NVIDIA RTX 4080M | CUDABackend (cuBLAS + cgo) | 🔲 Future |
| MacBook M4 | MetalBackend (MPS + cgo) | 🔲 Future |

## License

MIT
