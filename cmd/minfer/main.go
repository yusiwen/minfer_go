package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// Version info — injected via ldflags at build time.
// In the Makefile: go build -ldflags '-X "main.Version=$(VERSION)" ...'
var (
	Version    = "dev"
	CommitSHA  = "unknown"
	BuildTime  = "unknown"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "minfer",
		Short: "Minfer — Go LLM inference engine from scratch",
		Long: `Minfer is a from-scratch LLM (大语言模型) local inference engine written in Go.
It is designed as a learning project to deeply understand every layer of
LLM inference — tensor operations, GGUF format, tokenization, transformer
architecture, KV cache, and sampling.`,
	}

	// version subcommand
	rootCmd.AddCommand(&cobra.Command{
		Use:   "version",
		Short: "Print version information",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("minfer %s\n", Version)
			fmt.Printf("commit: %s\n", CommitSHA)
			fmt.Printf("built:  %s\n", BuildTime)
		},
	})

	// run subcommand
	runCmd := &cobra.Command{
		Use:   "run [model.gguf]",
		Short: "Run inference with a GGUF model",
		Args:  cobra.MaximumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			modelPath, _ := cmd.Flags().GetString("model")

			if modelPath == "" && len(args) > 0 {
				modelPath = args[0]
			}

			if modelPath == "" {
				fmt.Println("❌ No model specified. Use --model or pass as argument.")
				return
			}

			fmt.Printf("🔧 Minfer %s\n", Version)
			fmt.Printf("   Model:   %s\n", modelPath)
			fmt.Printf("\n⚠️  Model inference not yet implemented — coming in Phase 4-5.\n")
			fmt.Printf("   Phase 1 (Tensor + CPU backend) is verified via 'go test ./...'.\n")
		},
	}
	runCmd.Flags().StringP("model", "m", "", "Path to GGUF model file")
	rootCmd.AddCommand(runCmd)

	// pull subcommand (placeholder, implemented in Phase 6)
	pullCmd := &cobra.Command{
		Use:   "pull <model-ref>",
		Short: "Download a model from Hugging Face or Ollama registry",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("📥 Pull not yet implemented: %s\n", args[0])
			fmt.Println("   For now, download GGUF files manually from Hugging Face.")
		},
	}
	rootCmd.AddCommand(pullCmd)

	// list subcommand (placeholder)
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
