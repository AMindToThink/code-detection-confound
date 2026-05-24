"""Filter + per-language format HMCorp.

For every (code, contrast) row in the HMCorp JSONLs:
  - parse-check both fields with the language's native parser (reject row if either fails)
  - format both fields with the canonical formatter (reject if either format fails)

Writes two output files per (lang, split), sharing the same row set so the only
difference between training the raw baseline and training CodeGPTSensorBlack is the
formatter:
  data/hmcorp/{python,java}/{train,valid,test}.jsonl              (cleaned, NOT formatted)
  data/hmcorp/{python,java}/{train,valid,test}_formatted.jsonl    (cleaned + formatted)

Python: ast.parse + black.format_str, in-process, Pool(14).
Java:   javalang.parse + google-java-format. Java rows in HMCorp are method-level (not
        full compilation units), so they are wrapped in `class A_i { ... }` for parsing
        and formatting; the wrapper is stripped from the output. Pool(4) workers each
        invoke one JVM per ~1000-snippet batch to amortize ~500 ms startup.

Run:
  uv run --no-project --with black --with javalang --with pandas python -u scripts/15_format_hmcorp.py
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import black
import javalang
import javalang.parse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

JAR = str((C.ROOT / "external/google-java-format.jar").resolve())
HMCORP_DIR = C.ROOT / "external/CodeGPTSensor/dataset"
OUT_DIR = C.DATA / "hmcorp"

PY_WORKERS = 14
JAVA_WORKERS = 4
JAVA_CHUNK = 1000  # snippets per JVM invocation (amortizes ~500 ms startup)


# ----------------------- Python -----------------------

def py_parse_format(code: str) -> str | None:
    try:
        ast.parse(code)
    except Exception:
        return None
    try:
        return black.format_str(code, mode=black.Mode())
    except Exception:
        return None


def py_pair(row_pair: tuple[str, str]) -> tuple[str, str] | None:
    a, b = row_pair
    af = py_parse_format(a)
    if af is None:
        return None
    bf = py_parse_format(b)
    if bf is None:
        return None
    return af, bf


# ----------------------- Java -----------------------

def java_parse_ok(code: str) -> bool:
    try:
        javalang.parse.parse("class _W {\n" + code + "\n}")
        return True
    except Exception:
        return False


def strip_wrapper(wrapped: str) -> str | None:
    """Remove 'class A_xxxxxxxxxx { ... }' wrapper. Dedent body by 2 spaces.
    Returns None if the wrapper isn't recognizable (= format failure)."""
    lines = wrapped.splitlines()
    # find opening line ending with '{'
    start = 0
    while start < len(lines) and not lines[start].rstrip().endswith("{"):
        start += 1
    if start >= len(lines):
        return None
    body = lines[start + 1:]
    # strip trailing blank lines, then the closing '}'
    while body and body[-1].strip() == "":
        body.pop()
    if not body or body[-1].strip() != "}":
        return None
    body.pop()
    # dedent by 2 spaces (google-java-format uses 2-space indent inside classes)
    out = [(line[2:] if line.startswith("  ") else line) for line in body]
    return ("\n".join(out)).rstrip() + "\n"


def format_java_chunk(args: tuple[int, list[tuple[int, str]]]) -> dict[int, str]:
    """Wrap each snippet, run one JVM invocation over all files in the chunk, strip."""
    chunk_id, snippets = args
    out: dict[int, str] = {}
    with tempfile.TemporaryDirectory() as d:
        paths: list[str] = []
        for idx, code in snippets:
            p = Path(d) / f"A_{idx:010d}.java"
            p.write_text(f"class A_{idx:010d} {{\n{code}\n}}\n")
            paths.append(str(p))
        try:
            subprocess.run(
                ["java", "-jar", JAR, "-i", *paths],
                capture_output=True, text=True, timeout=900,
            )
        except subprocess.TimeoutExpired:
            return out
        for idx, _ in snippets:
            try:
                content = (Path(d) / f"A_{idx:010d}.java").read_text()
            except Exception:
                continue
            stripped = strip_wrapper(content)
            if stripped is not None:
                out[idx] = stripped
    return out


# ----------------------- per-split drivers -----------------------

def process_python(in_path: Path, out_raw: Path, out_fmt: Path) -> None:
    rows = [json.loads(l) for l in in_path.open()]
    pairs = [(r["code"], r["contrast"]) for r in rows]
    with Pool(PY_WORKERS) as P:
        results = P.map(py_pair, pairs, chunksize=200)
    kept = 0
    with out_raw.open("w") as fr, out_fmt.open("w") as ff:
        for r, res in zip(rows, results):
            if res is None:
                continue
            af, bf = res
            fr.write(json.dumps(r) + "\n")
            ff.write(json.dumps({**r, "code": af, "contrast": bf}) + "\n")
            kept += 1
    print(f"  python {in_path.name}: kept {kept}/{len(rows)} ({kept/len(rows):.3f})", flush=True)


def process_java(in_path: Path, out_raw: Path, out_fmt: Path) -> None:
    rows = [json.loads(l) for l in in_path.open()]
    # pre-filter via javalang on BOTH fields (cheap, in-process, parallel)
    with Pool(PY_WORKERS) as P:
        ok_code = P.map(java_parse_ok, [r["code"] for r in rows], chunksize=500)
        ok_contrast = P.map(java_parse_ok, [r["contrast"] for r in rows], chunksize=500)
    keep = [i for i in range(len(rows)) if ok_code[i] and ok_contrast[i]]
    print(f"  java {in_path.name}: javalang-ok on both -> {len(keep)}/{len(rows)}", flush=True)

    # collect snippets to format (each kept row contributes 2)
    items: list[tuple[int, str]] = []
    for i in keep:
        items.append((i * 2,     rows[i]["code"]))
        items.append((i * 2 + 1, rows[i]["contrast"]))
    chunks = [
        (c, items[c * JAVA_CHUNK:(c + 1) * JAVA_CHUNK])
        for c in range((len(items) + JAVA_CHUNK - 1) // JAVA_CHUNK)
    ]
    print(f"  java {in_path.name}: {len(items)} snippets in {len(chunks)} JVM batches", flush=True)
    with Pool(JAVA_WORKERS) as P:
        partial = P.map(format_java_chunk, chunks)
    fmt: dict[int, str] = {}
    for d in partial:
        fmt.update(d)

    kept = 0
    with out_raw.open("w") as fr, out_fmt.open("w") as ff:
        for i in keep:
            af = fmt.get(i * 2)
            bf = fmt.get(i * 2 + 1)
            if af is None or bf is None:
                continue
            r = rows[i]
            fr.write(json.dumps(r) + "\n")
            ff.write(json.dumps({**r, "code": af, "contrast": bf}) + "\n")
            kept += 1
    print(f"  java {in_path.name}: fully formatted {kept}/{len(rows)} ({kept/len(rows):.3f})",
          flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("python", "java"):
        (OUT_DIR / lang).mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            in_path = HMCORP_DIR / lang / f"{split}.jsonl"
            out_raw = OUT_DIR / lang / f"{split}.jsonl"
            out_fmt = OUT_DIR / lang / f"{split}_formatted.jsonl"
            print(f"[{lang}/{split}] -> {out_raw.name} + {out_fmt.name}", flush=True)
            (process_python if lang == "python" else process_java)(in_path, out_raw, out_fmt)


if __name__ == "__main__":
    main()
