from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parents[1]
pat=re.compile(r"\[[^\]]+\]\(([^)]+)\)")
broken=[]
for md in ROOT.rglob("*.md"):
    for target in pat.findall(md.read_text(encoding="utf-8")):
        target=target.split("#",1)[0]
        if not target or target.startswith(("http://","https://","mailto:")): continue
        if not (md.parent/target).resolve().exists():
            broken.append((md.relative_to(ROOT),target))
if broken:
    for a,b in broken: print(f"BROKEN {a}: {b}")
    sys.exit(1)
print("All internal Markdown links resolve.")
