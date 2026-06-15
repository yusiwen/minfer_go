package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/yusiwen/minfer/internal/backend/cpu"
	"github.com/yusiwen/minfer/internal/tensor"
)

// Version 信息，通过 ldflags 在构建时注入
// 在 Makefile 中：go build -ldflags '-X "main.Version=$(VERSION)" ...'
var (
	Version    = "dev"
	CommitSHA  = "unknown"
	BuildTime  = "unknown"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "minfer",
		Short: "Minfer — Go LLM inference engine from scratch",
		Long: `Minfer is a from-scratch LLM local inference engine written in Go.
It is designed as a learning project to deeply understand every layer of
LLM inference — tensor operations, GGUF format, tokenization, transformer
architecture, KV cache, and sampling.`,
	}

	// version 子命令
	rootCmd.AddCommand(&cobra.Command{
		Use:   "version",
		Short: "Print version information",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("minfer %s\n", Version)
			fmt.Printf("commit: %s\n", CommitSHA)
			fmt.Printf("built:  %s\n", BuildTime)
		},
	})

	// run 子命令
	runCmd := &cobra.Command{
		Use:   "run [model.gguf]",
		Short: "Run inference with a GGUF model",
		Args:  cobra.MaximumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			modelPath, _ := cmd.Flags().GetString("model")
			backendName, _ := cmd.Flags().GetString("backend")

			if modelPath == "" && len(args) > 0 {
				modelPath = args[0]
			}

			fmt.Printf("🔧 Minfer %s\n", Version)
			fmt.Printf("   Model:   %s\n", modelPath)
			fmt.Printf("   Backend: %s\n", backendName)

			if modelPath == "" {
				fmt.Println("❌ No model specified. Use --model or pass as argument.")
				return
			}

			// ---- Phase 1 test: verify CPUBackend works ----
			b := cpu.New()

			// 做一个最简单的 2×3 × 3×2 = 2×2 矩阵乘法来验证
			a := tensor.NewWithData([]float32{
				1, 2, 3,
				4, 5, 6,
			}, 2, 3)
			bTensor := tensor.NewWithData([]float32{
				7, 8,
				9, 10,
				11, 12,
			}, 3, 2)

			c := b.MatMul(a, bTensor)
			fmt.Printf("\n✅ MatMul test passed!\n")
			fmt.Printf("   A[2×3] × B[3×2] = C[2×2]:\n")
			fmt.Printf("   ⎡%.0f  %.0f⎤\n", c.Data[0], c.Data[1])
			fmt.Printf("   ⎣%.0f  %.0f⎦\n", c.Data[2], c.Data[3])
		},
	}
	runCmd.Flags().StringP("model", "m", "", "Path to GGUF model file")
	runCmd.Flags().StringP("backend", "b", "cpu", "Compute backend (cpu, cuda, metal)")
	rootCmd.AddCommand(runCmd)

	// pull 子命令（占位，后续实现）
	pullCmd := &cobra.Command{
		Use:   "pull [model-ref]",
		Short: "Download a model from Hugging Face or Ollama registry",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("📥 Pull not yet implemented: %s\n", args[0])
			fmt.Println("   For now, download GGUF files manually from Hugging Face.")
		},
	}
	rootCmd.AddCommand(pullCmd)

	// list 子命令（占位）
	rootCmd.AddCommand(&cobra.Command{
		Use:   "list",
		Short: "List downloaded models",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Println("📋 list: not yet implemented")
		},
	})

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
