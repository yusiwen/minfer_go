package registry

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// PullHF downloads a model from Hugging Face Hub.
//
// Ref format: "hf:owner/repo/filename.gguf"
// Example: "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf"
//
// The file is saved to ~/.cache/minfer/models/hf/{owner}/{repo}/{filename}.
// An alias is created with the last path segment of the repo as the name.
func PullHF(ref string) (string, error) {
	// Parse the ref: strip "hf:" prefix
	ref = strings.TrimPrefix(ref, "hf:")

	// Split into owner/repo/filename
	parts := strings.SplitN(ref, "/", 3)
	if len(parts) < 3 {
		return "", fmt.Errorf("registry: invalid HF ref %q, expected hf:owner/repo/filename.gguf", ref)
	}
	owner := parts[0]
	repo := parts[1]
	filename := parts[2]

	if !strings.HasSuffix(filename, ".gguf") {
		return "", fmt.Errorf("registry: HF filename must end with .gguf, got %q", filename)
	}

	// Destination path
	destDir := filepath.Join(modelCacheDir("hf"), owner, repo)
	if err := EnsureDir(destDir); err != nil {
		return "", fmt.Errorf("registry: creating cache dir: %w", err)
	}
	destPath := filepath.Join(destDir, filename)

	// Check if already cached
	if _, err := os.Stat(destPath); err == nil {
		// Already exists — just ensure alias
		aliasName := strings.TrimSuffix(filename, ".gguf")
		AddAlias(aliasName, "hf", destPath, "hf:"+ref)
		return destPath, nil
	}

	// Download
	url := fmt.Sprintf("https://huggingface.co/%s/%s/resolve/main/%s", owner, repo, filename)
	fmt.Printf("📥 Downloading %s ...\n", url)

	if err := downloadFile(url, destPath); err != nil {
		return "", fmt.Errorf("registry: downloading from HF: %w", err)
	}

	// Create alias
	aliasName := strings.TrimSuffix(filename, ".gguf")
	if err := AddAlias(aliasName, "hf", destPath, "hf:"+ref); err != nil {
		return "", fmt.Errorf("registry: saving alias: %w", err)
	}

	return destPath, nil
}

// downloadFile downloads a URL to a local file.
func downloadFile(url, destPath string) error {
	// Create temporary file in the same directory (atomic rename on success)
	tmpPath := destPath + ".tmp"

	out, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer out.Close()

	resp, err := http.Get(url)
	if err != nil {
		os.Remove(tmpPath)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		os.Remove(tmpPath)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	if _, err := io.Copy(out, resp.Body); err != nil {
		os.Remove(tmpPath)
		return err
	}

	if err := out.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}

	return os.Rename(tmpPath, destPath)
}
