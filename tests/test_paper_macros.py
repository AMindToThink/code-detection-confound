"""Guard the script->paper number contract (cf. CLAUDE.md paper-writing discipline):

1. Every \\res... macro referenced in paper/paper.tex is defined in paper/macros.tex.
2. paper.tex contains no hand-typed decimal result numbers in prose (digits like 0.73);
   results must come through macros. (Integers in margins/geometry and the year are
   allowed; we scan only outside the preamble and ignore \\res tokens.)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "paper.tex"
MACROS = ROOT / "paper" / "macros.tex"


def test_all_referenced_macros_are_defined():
    defined = set(re.findall(r"\\newcommand\{\\(res[A-Za-z]+)\}", MACROS.read_text()))
    used = set(re.findall(r"\\(res[A-Za-z]+)\b", PAPER.read_text()))
    missing = used - defined
    assert not missing, f"macros used in paper.tex but undefined in macros.tex: {sorted(missing)}"


def test_no_handtyped_decimal_results_in_prose():
    text = PAPER.read_text()
    body = text.split(r"\begin{document}", 1)[-1]      # skip preamble (geometry etc.)
    body = re.sub(r"\\res[A-Za-z]+", "", body)          # macros are the allowed source
    body = re.sub(r"\\includegraphics\[[^\]]*\]", "", body)  # widths like 0.8\linewidth
    body = re.sub(r"%.*", "", body)                     # comments
    # ignore decimals embedded in identifiers/version tokens (e.g. gpt-5.4-nano)
    decimals = re.findall(r"(?<![\w-])\d+\.\d+(?![\w-])", body)
    assert not decimals, f"hand-typed decimal numbers in prose (use macros): {decimals}"
