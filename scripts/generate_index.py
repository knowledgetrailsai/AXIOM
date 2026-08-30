from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")):
    docs=sorted(folder.glob("*.md"))
    if not docs: continue
    print(f"## {folder.name}")
    for doc in docs:
        title=next((x[2:] for x in doc.read_text(encoding="utf-8").splitlines() if x.startswith("# ")),doc.stem)
        print(f"- [{title}]({doc.relative_to(ROOT).as_posix()})")
    print()
