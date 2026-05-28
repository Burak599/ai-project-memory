# code_layers/code_input_layer.py

import os
from typing import List, Dict

# Taranmayacak klasörler
EXCLUDED_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", ".env",
    "node_modules", ".idea", ".vscode", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "eggs",
    ".eggs", "*.egg-info",
}

# Taranacak uzantılar
INCLUDED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".r", ".m",
}

# Atlanacak dosyalar
EXCLUDED_FILES = {
    "__init__.py",         # genellikle boş
    "setup.py",
    "conftest.py",
}

# Tek dosya max karakter (çok büyük dosyaları kırp)
MAX_FILE_CHARS = 30000


class CodeInputLayer:
    """
    Verilen proje klasörünü tarar, kod dosyalarını okur ve
    sıralı bir liste halinde döner.

    Sıralama mantığı:
    1. Ana giriş noktaları önce (main.py, app.py, index.py vb.)
    2. Sonra klasör derinliğine göre (sığ önce)
    3. Sonra alfabetik
    """

    ENTRY_POINTS = {
        "main.py", "app.py", "index.py", "run.py",
        "server.py", "cli.py", "manage.py", "start.py",
    }

    def __init__(
        self,
        max_file_chars: int = MAX_FILE_CHARS,
        included_extensions: set = None,
    ):
        self.max_file_chars = max_file_chars
        self.included_extensions = included_extensions or INCLUDED_EXTENSIONS

    def scan(self, project_path: str) -> List[Dict]:
        """
        Klasörü tarar ve sıralı dosya listesi döner.

        Args:
            project_path: Proje kök klasörünün yolu

        Returns:
            [
                {
                    "index":     1,               # sıra numarası
                    "path":      "layers/foo.py", # proje köküne göre relatif
                    "abs_path":  "/full/path...", # mutlak yol
                    "extension": ".py",
                    "size_chars": 1234,
                    "content":   "...",           # dosya içeriği (kırpılmış olabilir)
                    "truncated": False,           # içerik kırpıldı mı?
                },
                ...
            ]
        """
        if not os.path.isdir(project_path):
            raise ValueError(f"[CodeInputLayer] Geçersiz klasör: {project_path}")

        raw_files = self._collect_files(project_path)
        sorted_files = self._sort_files(raw_files)

        results = []
        for i, file_info in enumerate(sorted_files, start=1):
            content, truncated = self._read_file(file_info["abs_path"])
            results.append({
                "index":      i,
                "path":       file_info["rel_path"],
                "abs_path":   file_info["abs_path"],
                "extension":  file_info["extension"],
                "size_chars": len(content),
                "content":    content,
                "truncated":  truncated,
            })

        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_files(self, project_path: str) -> List[Dict]:
        """Klasörü özyinelemeli tarar, uygun dosyaları toplar."""
        files = []
        project_path = os.path.abspath(project_path)

        for root, dirs, filenames in os.walk(project_path):
            # Hariç tutulan klasörleri atla (yerinde filtrele)
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDED_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.included_extensions:
                    continue
                if filename in EXCLUDED_FILES:
                    continue

                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, project_path)
                depth = rel_path.count(os.sep)

                files.append({
                    "abs_path":  abs_path,
                    "rel_path":  rel_path,
                    "filename":  filename,
                    "extension": ext,
                    "depth":     depth,
                })

        return files

    def _sort_files(self, files: List[Dict]) -> List[Dict]:
        """
        Sıralama:
        0 → giriş noktaları (main.py vb.)
        1 → diğerleri (derinlik, sonra alfabetik)
        """
        def sort_key(f):
            is_entry = 0 if f["filename"] in self.ENTRY_POINTS else 1
            return (is_entry, f["depth"], f["rel_path"])

        return sorted(files, key=sort_key)

    def _read_file(self, abs_path: str):
        """
        Dosyayı okur. Çok büyükse kırpar.

        Returns:
            (content: str, truncated: bool)
        """
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return f"[OKUNAMADI: {e}]", False

        if len(content) > self.max_file_chars:
            content = content[: self.max_file_chars]
            content += f"\n\n... [DOSYA KISALTI LDI — ilk {self.max_file_chars} karakter gösteriliyor]"
            return content, True

        return content, False