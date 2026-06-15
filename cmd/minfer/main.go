package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/yusiwen/minfer/internal/backend/cpu"
	"github.com/yusiwen/minfer/internal/gguf"
	"github.com/yusiwen/minfer/internal/infer"
	"github.com/yusiwen/minfer/internal/model"
	"github.com/yusiwen/minfer/internal/registry"
	"github.com/yusiwen/minfer/internal/tokenizer"
)

// Version info — injected via ldflags at build time.
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
		Use:   "run <model> [prompt]",
		Short: "Run inference with a GGUF model",
		Long: `Run inference with a GGUF model file.

Examples:
  minfer run qwen2.5-0.5b-instruct-q4_0.gguf "Hello, world!"
  minfer run --prompt "What is Rust?" model.gguf
`,
		Args: cobra.MinimumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			modelPath := args[0]
			prompt, _ := cmd.Flags().GetString("prompt")

			// If no prompt flag, use remaining args as prompt
			if prompt == "" && len(args) > 1 {
				prompt = args[1]
			}
			if prompt == "" {
				prompt = "The meaning of life is"
			}

			temperature, _ := cmd.Flags().GetFloat64("temperature")
			maxTokens, _ := cmd.Flags().GetInt("max-tokens")

			fmt.Printf("🔧 Minfer %s\n", Version)
			fmt.Printf("   Model:  %s\n", modelPath)
			fmt.Printf("   Prompt: %q\n", prompt)
			fmt.Printf("\n📂 Opening model file...\n")

			// Step 1: Open GGUF file
			f, err := os.Open(modelPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Error opening model: %v\n", err)
				os.Exit(1)
			}
			defer f.Close()

			reader, err := gguf.Open(f)
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Error reading GGUF header: %v\n", err)
				os.Exit(1)
			}

			fmt.Printf("   Architecture: %s\n", reader.Architecture())
			fmt.Printf("   Tensors:      %d\n", reader.TensorCount)

			// Step 2: Load model config
			cfg, err := reader.LoadConfig()
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Error loading config: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("   Hidden dim:   %d\n", cfg.HiddenDim)
			fmt.Printf("   Layers:       %d\n", cfg.NumLayers)
			fmt.Printf("   Heads:        %d\n", cfg.NumHeads)
			fmt.Printf("   Vocab size:   %d\n", cfg.VocabSize)

			// Step 3: Load model weights
			fmt.Printf("\n📦 Loading weights...\n")
			backend := cpu.New()
			m, err := model.LoadModel(reader, cfg, backend)
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Error loading model: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("   ✅ Model loaded (%d layers)\n", cfg.NumLayers)

			// Step 4: Create tokenizer
			fmt.Printf("\n🔤 Loading tokenizer...\n")
			tok, err := tokenizer.LoadFromGGUF(reader)
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Error loading tokenizer: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("   ✅ Tokenizer ready (vocab: %d)\n", tok.VocabSize())

			// Step 5: Run inference
			fmt.Printf("\n🤖 Generating...\n")
			fmt.Printf("   ─────────────────────────────────────────────\n")

			samplerCfg := infer.DefaultSamplerConfig()
			if temperature > 0 {
				samplerCfg.Temperature = float32(temperature)
			}

			engine := &infer.Engine{
				Model:         modelAdapter{m},
				Tokenizer:     tok,
				SamplerConfig: samplerCfg,
				MaxTokens:     maxTokens,
			}

			output, err := engine.Generate(prompt)
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Generation error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("   %s\n", output)
			fmt.Printf("   ─────────────────────────────────────────────\n")
			fmt.Printf("✅ Generation complete\n")
		},
	}
	runCmd.Flags().StringP("prompt", "p", "", "Input prompt")
	runCmd.Flags().Float64P("temperature", "t", 0, "Sampling temperature (0=greedy, default=0.7)")
	runCmd.Flags().IntP("max-tokens", "n", 512, "Maximum tokens to generate")
	rootCmd.AddCommand(runCmd)

	// pull subcommand
	pullCmd := &cobra.Command{
		Use:   "pull <model-ref>",
		Short: "Download a model from Hugging Face or Ollama registry",
		Long: `Download a model from Hugging Face Hub or the Ollama registry.

Examples:
  minfer pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
  minfer pull ollama:qwen2.5:0.5b
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

// modelAdapter wraps *model.Model to implement infer.ModelForwarder.
type modelAdapter struct {
	m *model.Model
}

func (a modelAdapter) Forward(tokens []int, startPos int) ([]float32, error) {
	return model.ForwardAdapter(a.m, tokens, startPos)
}
