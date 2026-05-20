"""Surface style features for Python source. Used for (a) the mediation ablation
(does detector score reduce to style features?) and (b) the difficulty->style check
(do harder-problem human solutions actually look more 'expert'?).

All features are regex/heuristic based so they work on code that does not parse
(LLM output and terse competitive code frequently won't).
"""
from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE_HINT = re.compile(r"(:\s*[A-Za-z_][\w\[\], .]*\s*[=)\n]|->\s*[A-Za-z_])")
_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
    "pass", "raise", "return", "try", "while", "with", "yield", "print", "input",
    "range", "len", "int", "str", "list", "map", "sum", "min", "max", "sorted",
}


def style_features(code: str) -> dict[str, float]:
    lines = code.splitlines()
    n_lines = max(len(lines), 1)
    nonblank = [ln for ln in lines if ln.strip()]
    comment_lines = sum(1 for ln in lines if ln.strip().startswith("#"))
    blank_lines = sum(1 for ln in lines if not ln.strip())

    # identifiers that aren't keywords/builtins -> author-chosen names
    idents = [t for t in _IDENT.findall(code) if t not in _KEYWORDS]
    mean_ident_len = sum(len(t) for t in idents) / max(len(idents), 1)
    single_char = sum(1 for t in idents if len(t) == 1) / max(len(idents), 1)

    has_docstring = 1.0 if re.search(r'(""".*?"""|\'\'\'.*?\'\'\')', code, re.DOTALL) else 0.0
    has_type_hints = 1.0 if _TYPE_HINT.search(code) else 0.0
    has_main_guard = 1.0 if 'if __name__' in code else 0.0
    n_def = len(re.findall(r"\bdef\b", code))

    return {
        "comment_density": comment_lines / n_lines,
        "blank_line_ratio": blank_lines / n_lines,
        "mean_identifier_len": mean_ident_len,
        "single_char_ident_frac": single_char,
        "mean_line_len": sum(len(ln) for ln in nonblank) / max(len(nonblank), 1),
        "has_docstring": has_docstring,
        "has_type_hints": has_type_hints,
        "has_main_guard": has_main_guard,
        "n_defs": float(n_def),
        "n_lines": float(n_lines),
        "code_chars": float(len(code)),
    }


STYLE_COLS = list(style_features("x=1").keys())
