import ast
from pathlib import Path
import config
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)

def _make(path, language, kind, symbol, start, end, body):
    return {
        "text": f"# File: {path}\n# Symbol: {symbol}\n{body}",
        "path": path,
        "symbol": symbol,
        "kind": kind,
        "start_line": start,
        "end_line": end,
        "language": language,
    }


def _windows(lines, first_line, max_chars, overlap):
    """Yield (start_line, end_line, text) windows of at most max_chars."""
    start, n = 0, len(lines)
    while start < n:
        size, end = 0, start
        while end < n and (end == start or size + len(lines[end]) + 1 <= max_chars):
            size += len(lines[end]) + 1
            end += 1
        yield first_line + start, first_line + end - 1, "\n".join(lines[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)


def _emit(path, language, kind, symbol, first_line, lines):
    """One chunk if it fits, otherwise several overlapping parts."""
    body = "\n".join(lines)
    if len(body) <= config.MAX_CHUNK_CHARS:
        return [_make(path, language, kind, symbol, first_line, first_line + len(lines) - 1, body)]
    return [
        _make(path, language, kind, f"{symbol} (part {i})", s, e, t)
        for i, (s, e, t) in enumerate(
            _windows(lines, first_line, config.MAX_CHUNK_CHARS, config.OVERLAP_LINES), 1)
    ]


def _span(node):
    """Start line includes decorators; end line is inclusive."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno


def chunk_text(path, source, language, kind="doc"):
    lines = source.splitlines()
    if not "".join(lines).strip():
        return []
    return _emit(path, language, kind, "<file>", 1, lines)


def chunk_python(path, source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text(path, source, "python", kind="file")

    lines = source.splitlines()
    chunks = []
    defs = (*FUNC, ast.ClassDef)

    # 1. Module header: everything before the first def/class
    first_def = next((_span(n)[0] for n in tree.body if isinstance(n, defs)), None)
    header = lines[: (first_def - 1) if first_def else len(lines)]
    if "".join(header).strip():
        chunks += _emit(path, "python", "module", "<module>", 1, header)

    # 2. Functions, classes, methods
    rest = []  # top-level statements that appear after the first def
    for node in tree.body:
        if isinstance(node, FUNC):
            s, e = _span(node)
            chunks += _emit(path, "python", "function", node.name, s, lines[s - 1:e])
        elif isinstance(node, ast.ClassDef):
            s, e = _span(node)
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            doc = (ast.get_docstring(node) or "")[: config.MAX_CHUNK_CHARS // 2]
            methods = [n.name for n in node.body if isinstance(n, FUNC)]
            overview = f'class {node.name}({bases}):\n"""{doc}"""\nMethods: {", ".join(methods)}'
            chunks.append(_make(path, "python", "class", node.name, s, e, overview))
            for sub in node.body:
                if isinstance(sub, FUNC):
                    ms, me = _span(sub)
                    chunks += _emit(path, "python", "method", f"{node.name}.{sub.name}",
                                    ms, lines[ms - 1:me])
        elif first_def and node.lineno >= first_def:
            rest.append(node)

    # 3. Leftover module-level code after the first def (if/else blocks, constants...)
    if rest:
        s, e = rest[0].lineno, rest[-1].end_lineno
        chunks += _emit(path, "python", "module", "<module-level>", s, lines[s - 1:e])
    return chunks


def chunk_file(root, rel_path):
    source = (Path(root) / rel_path).read_text(encoding="utf-8", errors="ignore")
    language = config.EXTENSIONS[Path(rel_path).suffix]
    if language == "python":
        return chunk_python(rel_path, source)
    return chunk_text(rel_path, source, language)