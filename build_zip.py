"""Build a clean submission zip- excludes secrets, venv, cache, local DB.
Usage: python build_zip.py
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "aardvark.zip"

EXCLUDE_DIRS = {".venv", "data", "__pycache__", ".pytest_cache",
                ".git", ".idea", ".vscode", "node_modules"}
EXCLUDE_FILES = {".env", ".DS_Store", "build_zip.py", "aardvark.zip"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite", ".sqlite3"}


def should_skip(path: Path) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def main():
    if OUT.exists():
        OUT.unlink()
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            if p.is_dir():
                continue
            if should_skip(p):
                continue
            arc = p.relative_to(ROOT)
            zf.write(p, arcname=str(Path("deep-research-agent") / arc))
            count += 1
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.name}  ({count} files, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
