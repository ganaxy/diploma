import ast, io, os, sys, tokenize
from pathlib import Path

SKIP_DIRS  = {"__pycache__", ".git", "fb_profile", "WasmTtsEngine", ".ipynb_checkpoints"}
SKIP_SELF  = Path(__file__).resolve()


def docstring_line_ranges(source: str):
    ranges = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ranges
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ds = body[0]
                ranges.append((ds.lineno, ds.end_lineno))
    return ranges


def strip_file(source: str) -> str:
    ds_ranges = docstring_line_ranges(source)
    docstring_lines = set()
    for start, end in ds_ranges:
        docstring_lines.update(range(start, end + 1))

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        tokens = []

    comment_lines = set()
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])

    remove_lines = docstring_lines | comment_lines

    out_lines = []
    prev_blank = False
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in remove_lines:
            continue
        stripped = line.strip()
        if stripped == "":
            if not prev_blank:
                out_lines.append("")
            prev_blank = True
        else:
            out_lines.append(line.rstrip())
            prev_blank = False

    return "\n".join(out_lines).strip() + "\n"


def run(root: str):
    root_path = Path(root)
    files = []
    for dp, dns, fns in os.walk(root_path):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith(".py"):
                fp = Path(dp) / fn
                if fp.resolve() != SKIP_SELF:
                    files.append(fp)

    print(f"Stripping comments from {len(files)} Python files...\n")
    ok = err = 0
    for fp in sorted(files):
        rel = fp.relative_to(root_path)
        try:
            orig = fp.read_text(encoding="utf-8", errors="replace")
            clean = strip_file(orig)
            removed = orig.count("\n") - clean.count("\n")
            fp.write_text(clean, encoding="utf-8")
            tag = "   " if removed == 0 else f"{removed:+4d}"
            print(f"  {tag}  {rel}")
            ok += 1
        except Exception as e:
            print(f"  ERR  {rel}  — {e}")
            err += 1
    print(f"\n{ok} files cleaned, {err} errors.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent))
