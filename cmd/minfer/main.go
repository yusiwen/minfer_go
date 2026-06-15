package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/yusiwen/minfer/internal/registry"
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

	// pull subcommand
	pullCmd := &cobra.Command{
		Use:   "pull <model-ref>",
		Short: "Download a model from Hugging Face or Ollama registry",
		Long: `Download a model from Hugging Face Hub or the Ollama registry.

Examples:
  minfer pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
  minfer pull ollama:qwen2.5:0.5b
  minfer pull ollama:llama3.2:1b
`,
		Args: cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			ref := args[0]

			var path string
			var err error

			if len(ref) > 3 && ref[:3] == "hf:" {
				path, err = registry.PullHF(ref)
			} else if len(ref) > 7 && ref[:7] == "ollama:" {
				path, err = registry.PullOllama(ref)
			} else {
				fmt.Println("❌ Unknown registry. Use hf: or ollama: prefix.")
				fmt.Println("   Example: minfer pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf")
				return
			}

			if err != nil {
				fmt.Printf("❌ Pull failed: %v\n", err)
				return
			}
			fmt.Printf("✅ Model saved to: %s\n", path)
			fmt.Println("   Run: minfer run <path>")
		},
	}
	rootCmd.AddCommand(pullCmd)

	// list subcommand
	rootCmd.AddCommand(&cobra.Command{
		Use:   "list",
		Short: "List downloaded models",
		Run: func(cmd *cobra.Command, args []string) {
			entries, err := registry.ListModels()
			if err != nil {
				fmt.Printf("❌ Error listing models: %v\n", err)
				return
			}
			if len(entries) == 0 {
				fmt.Println("📋 No models downloaded yet.")
				fmt.Println("   Use 'minfer pull' to download a model.")
				return
			}
			fmt.Println("📋 Downloaded models:")
			for _, e := range entries {
				fmt.Printf("   %-20s  %s\n", e.Name, e.Path)
			}
		},
	})

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
