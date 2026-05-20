"""Extract pre-LLM-era source code from renowned engineers' repositories.

Motivation (Matthew's idea): code written *before* LLMs existed is ironclad-human
ground truth (zero possibility of AI contamination), and these authors sit at the very
top of the skill distribution with no proxy needed. If detectors flag their code as AI,
that is a direct, vivid demonstration of the skill -> "AI" confound.

We pin specific commits from the pre-LLM era (<= 2015; mostly 2005-2013) and pull files
strongly attributable to the named engineer. Each file is split into ~50-line windows
to yield several samples. Clones go to data/repos/ (gitignored) and are removed after.

Output: data/legendary_code.parquet  (schema-compatible; species='human', dataset='legendary')
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

import pandas as pd

REPOS_DIR = C.DATA / "repos"
REPOS_DIR.mkdir(exist_ok=True)
WINDOW = 50          # lines per sample window
MIN_WINDOW_LINES = 18
MAX_WINDOWS_PER_AUTHOR = 40

# (author, year_label, repo_url, ref-or-DATE:yyyy-mm-dd, [file hints], lang, ext)
# A ref is a SHA/tag; "DATE:..." resolves to the last commit before that date on HEAD.
SPECS = [
    # The INITIAL git commit (2005) — written entirely by Linus Torvalds. Iconic SHA.
    ("Linus Torvalds", "2005", "https://github.com/git/git",
     "e83c5163316f89bfbde7d9ab23ca2e25604af290",
     ["read-cache.c", "cache.h", "init-db.c", "update-cache.c", "show-diff.c"], "c", ".c"),
    # Redis early (2012) — Salvatore Sanfilippo (antirez) wrote the core.
    ("Salvatore Sanfilippo", "2012", "https://github.com/redis/redis",
     "DATE:2012-06-01",
     ["src/redis.c", "src/dict.c", "src/t_string.c", "src/object.c", "src/networking.c"], "c", ".c"),
    # SQLite public mirror, 2013 — D. Richard Hipp.
    ("D. Richard Hipp", "2013", "https://github.com/sqlite/sqlite",
     "DATE:2013-06-01", ["src/btree.c", "src/vdbe.c", "src/select.c", "src/where.c"], "c", ".c"),
    # CPython standard library (2012) — Guido van Rossum / core devs.
    ("CPython core", "2012", "https://github.com/python/cpython",
     "v3.3.0", ["Lib/argparse.py", "Lib/functools.py", "Lib/heapq.py", "Lib/pprint.py"], "python", ".py"),
]


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! {' '.join(cmd[:3])}... -> {r.stderr.strip()[:160]}")
    return r.returncode == 0


def resolve_ref(dest: Path, ref: str) -> str | None:
    if not ref.startswith("DATE:"):
        return ref
    date = ref.split(":", 1)[1]
    r = subprocess.run(["git", "rev-list", "-1", f"--before={date}", "HEAD"],
                       cwd=dest, capture_output=True, text=True)
    sha = r.stdout.strip()
    return sha or None


def windows(text: str) -> list[str]:
    lines = text.splitlines()
    out = []
    for i in range(0, len(lines), WINDOW):
        chunk = lines[i:i + WINDOW]
        if sum(1 for ln in chunk if ln.strip()) >= MIN_WINDOW_LINES:
            out.append("\n".join(chunk))
    return out


def main() -> None:
    rows = []
    for author, year, url, ref, files, lang, ext in SPECS:
        name = url.rstrip("/").split("/")[-1]
        dest = REPOS_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        print(f"== {author} ({year}) {url}@{ref[:16]}")
        if not run(["git", "clone", "--quiet", "--no-checkout", "--filter=blob:none", url, str(dest)]):
            print("   clone failed; skipping")
            continue
        sha = resolve_ref(dest, ref)
        if sha is None or not run(["git", "checkout", "--quiet", sha], cwd=dest):
            print("   checkout failed; skipping")
            shutil.rmtree(dest, ignore_errors=True)
            continue
        # file-glob fallback if pinned files are absent at this ref
        present = [r for r in files if (dest / r).exists()]
        if not present:
            present = [str(p.relative_to(dest)) for p in sorted(dest.rglob(f"*{ext}"))[:5]]
            print(f"   pinned files absent; globbed {len(present)} {ext} files")
        n_auth = 0
        for rel in present:
            fp = dest / rel
            if not fp.exists():
                print(f"   missing {rel}")
                continue
            try:
                text = fp.read_text(errors="ignore")
            except Exception:
                continue
            for k, win in enumerate(windows(text)):
                if n_auth >= MAX_WINDOWS_PER_AUTHOR:
                    break
                rows.append({
                    "sample_id": f"LEG::{author.replace(' ', '_')}::{name}::{rel.replace('/', '_')}::{k}",
                    "problem_id": f"leg_{name}", "band": "legendary",
                    "problem_rating": None, "species": "human",
                    "model": author, "condition": f"prellm_{year}",
                    "code": win, "skill_source": "renowned_engineer_prellm",
                    "dataset": "legendary", "author": author, "lang": lang, "year": year,
                })
                n_auth += 1
        print(f"   {n_auth} windows")
        shutil.rmtree(dest, ignore_errors=True)   # free disk immediately

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no legendary code extracted — check network/commits")
    out = C.DATA / "legendary_code.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(df)} samples")
    print(df.groupby('author').size())


if __name__ == "__main__":
    main()
