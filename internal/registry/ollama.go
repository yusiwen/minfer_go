package registry

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// ollamaManifest represents the OCI manifest returned by the Ollama registry.
type ollamaManifest struct {
	SchemaVersion int                `json:"schemaVersion"`
	MediaType     string             `json:"mediaType"`
	Layers        []ollamaManifestLayer `json:"layers"`
}

type ollamaManifestLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

// PullOllama downloads a model from the Ollama registry.
//
// Ref format: "ollama:name:tag" or "ollama:name" (default tag: latest)
// Example: "ollama:qwen2.5:0.5b"
//
// The Ollama registry uses the OCI Distribution protocol (same as Docker Hub).
// A model consists of multiple blobs; we only download the GGUF weight blob
// (mediaType = "application/vnd.ollama.image.model").
//
// The file is saved to ~/.cache/minfer/models/ollama/{name}/{tag}.gguf.
func PullOllama(ref string) (string, error) {
	ref = strings.TrimPrefix(ref, "ollama:")

	// Split into name and tag
	parts := strings.SplitN(ref, ":", 2)
	name := parts[0]
	tag := "latest"
	if len(parts) > 1 {
		tag = parts[1]
	}

	// Destination
	destDir := filepath.Join(modelCacheDir("ollama"), name)
	if err := EnsureDir(destDir); err != nil {
		return "", fmt.Errorf("registry: creating cache dir: %w", err)
	}
	destPath := filepath.Join(destDir, tag+".gguf")

	// Check cache
	if _, err := os.Stat(destPath); err == nil {
		aliasName := name + ":" + tag
		AddAlias(aliasName, "ollama", destPath, "ollama:"+ref)
		return destPath, nil
	}

	// Step 1: Fetch the manifest
	manifestURL := fmt.Sprintf("https://registry.ollama.ai/v2/library/%s/manifests/%s", name, tag)
	fmt.Printf("📥 Fetching manifest: %s ...\n", manifestURL)

	req, err := http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		return "", fmt.Errorf("registry: creating manifest request: %w", err)
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("registry: fetching manifest: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("registry: manifest HTTP %d", resp.StatusCode)
	}

	var manifest ollamaManifest
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return "", fmt.Errorf("registry: decoding manifest: %w", err)
	}

	// Step 2: Find the GGUF model blob
	var modelBlob *ollamaManifestLayer
	for _, layer := range manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			modelBlob = &layer
			break
		}
	}
	if modelBlob == nil {
		return "", fmt.Errorf("registry: no model blob found in manifest for %s:%s", name, tag)
	}

	// Step 3: Download the blob
	blobURL := fmt.Sprintf("https://registry.ollama.ai/v2/library/%s/blobs/%s", name, modelBlob.Digest)
	fmt.Printf("📥 Downloading blob (%s, %d bytes) ...\n", modelBlob.Digest[:20], modelBlob.Size)

	tmpPath := destPath + ".tmp"
	out, err := os.Create(tmpPath)
	if err != nil {
		return "", fmt.Errorf("registry: creating temp file: %w", err)
	}

	blobResp, err := http.Get(blobURL)
	if err != nil {
		out.Close()
		os.Remove(tmpPath)
		return "", fmt.Errorf("registry: downloading blob: %w", err)
	}
	defer blobResp.Body.Close()

	if blobResp.StatusCode != http.StatusOK {
		out.Close()
		os.Remove(tmpPath)
		return "", fmt.Errorf("registry: blob HTTP %d", blobResp.StatusCode)
	}

	if _, err := io.Copy(out, blobResp.Body); err != nil {
		out.Close()
		os.Remove(tmpPath)
		return "", fmt.Errorf("registry: copying blob data: %w", err)
	}
	out.Close()

	if err := os.Rename(tmpPath, destPath); err != nil {
		return "", fmt.Errorf("registry: renaming temp file: %w", err)
	}

	// Create alias
	aliasName := name + ":" + tag
	if err := AddAlias(aliasName, "ollama", destPath, "ollama:"+ref); err != nil {
		return "", fmt.Errorf("registry: saving alias: %w", err)
	}

	return destPath, nil
}
