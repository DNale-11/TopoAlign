import ast, pathlib
p = pathlib.Path(r"F:\programme\nxy\Cell registration\napari-cell-registration\src\napari_cell_registration\_widget.py")
src = p.read_text(encoding="utf-8")
try:
    ast.parse(src)
    print(f"Syntax OK, total lines: {src.count(chr(10))}")
except SyntaxError as e:
    print(f"SyntaxError at line {e.lineno}: {e.msg}")
    lines = src.splitlines()
    start = max(0, e.lineno - 3)
    end = min(len(lines), e.lineno + 2)
    for i, ln in enumerate(lines[start:end], start=start+1):
        marker = ">>>" if i == e.lineno else "   "
        print(f"{marker} {i:4d}: {ln}")
