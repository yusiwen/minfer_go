// Package registry handles downloading models from Hugging Face Hub
// and the Ollama registry, managing the local model cache.
package registry

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// CacheDir returns the base cache directory (~/.cache/minfer).
func CacheDir() string {
	if v := os.Getenv("MINFER_CACHE_DIR"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "minfer")
}

// modelCacheDir returns the directory for a specific model source.
func modelCacheDir(source string) string {
	return filepath.Join(CacheDir(), "models", source)
}

// EnsureDir creates a directory if it doesn't exist.
func EnsureDir(path string) error {
	return os.MkdirAll(path, 0755)
}

// AliasEntry stores a short name mapping to a model path.
type AliasEntry struct {
	Name     string `json:"name"`
	Source   string `json:"source"`   // "hf" or "ollama"
	Path     string `json:"path"`     // local file path
	Original string `json:"original"` // original pull reference
}

// Aliases represents the alias file contents.
type Aliases struct {
	Entries []AliasEntry `json:"entries"`
}

// aliasesPath returns the path to the aliases JSON file.
func aliasesPath() string {
	return filepath.Join(CacheDir(), "aliases", "aliases.json")
}

// LoadAliases reads the alias file.
func LoadAliases() (*Aliases, error) {
	path := aliasesPath()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &Aliases{}, nil
		}
		return nil, err
	}
	var a Aliases
	if err := json.Unmarshal(data, &a); err != nil {
		return &Aliases{}, nil // corrupt file → start fresh
	}
	return &a, nil
}

// SaveAliases writes the alias file.
func SaveAliases(a *Aliases) error {
	path := aliasesPath()
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return err
	}
	data, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// AddAlias adds or updates an alias entry.
func AddAlias(name, source, path, original string) error {
	a, err := LoadAliases()
	if err != nil {
		return err
	}
	// Update existing or append
	for i := range a.Entries {
		if a.Entries[i].Name == name {
			a.Entries[i].Source = source
			a.Entries[i].Path = path
			a.Entries[i].Original = original
			return SaveAliases(a)
		}
	}
	a.Entries = append(a.Entries, AliasEntry{
		Name: name, Source: source, Path: path, Original: original,
	})
	return SaveAliases(a)
}

// ResolveAlias looks up a short name in the alias file.
func ResolveAlias(name string) (*AliasEntry, bool) {
	a, err := LoadAliases()
	if err != nil {
		return nil, false
	}
	for _, e := range a.Entries {
		if e.Name == name {
			return &e, true
		}
	}
	return nil, false
}

// ListModels returns all cached model entries.
func ListModels() ([]AliasEntry, error) {
	a, err := LoadAliases()
	if err != nil {
		return nil, err
	}
	return a.Entries, nil
}
