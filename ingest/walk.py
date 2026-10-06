import os
from pathlib import Path
import pathspec
import config

def walk_repo(root) -> list[str]:
    """Return repo-relative POSIX paths of files worth indexing."""
    root = Path(root)
    gi = root / ".gitignore"
    spec = pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text().splitlines() if gi.exists() else [])

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.SKIP_DIRS]
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root)
            rel_posix = rel.as_posix()
            if full.suffix not in config.EXTENSIONS or name in config.SKIP_FILES:
                continue
            if not config.INCLUDE_TESTS and any(p in config.TEST_DIRS for p in rel.parts[:-1]):
                continue
            if spec.match_file(rel_posix):
                continue
            if full.stat().st_size > config.MAX_FILE_BYTES:
                continue
            found.append(rel_posix)

    return sorted(found)