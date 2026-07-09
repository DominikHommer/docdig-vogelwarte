"""Schema-based column assignment for the banding forms.

The forms are PRINTED — the column order is fixed. Instead of guessing each
column independently from its header text (fragile: header OCR fails on
several columns per page, and short keywords used to substring-match into
the wrong column, which is how "Bague" ended up as UI column 5), we align
the extracted columns against a predefined schema:

- Every (column, role) pair gets a score from fuzzy header matching plus a
  width prior (e.g. Espèce is by far the widest column).
- A monotone sequence alignment (Needleman-Wunsch style dynamic programming)
  picks the best assignment that PRESERVES THE PRINTED ORDER — a later
  column can never be assigned an earlier role, each role at most once.
- Weak evidence is handled by the alignment: a column with an unreadable
  header still gets its role when it sits between two confidently matched
  anchors and its width fits.

The schema lives in ``config/column_schema.json`` and is user-editable
("Spalten vordefinieren"): keywords, expected width shares and the expected
order can be adapted per form type without touching code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz
from rapidfuzz import utils as fuzz_utils


DEFAULT_SCHEMA_PATH = Path("./config/column_schema.json")
# User edits from the in-app schema editor land here (gitignored) — the
# canonical default file above stays pristine; "Standard" deletes this file.
CUSTOM_SCHEMA_PATH = Path("./config/column_schema_custom.json")

ROLE_TO_FLAG = {
    "batch": "is_batch_column",
    "species": "is_species_column",
    "sexe": "is_sexe_column",
    "age": "is_age_column",
    "jour_mois": "is_jour-mois_column",
    "heure": "is_heure_column",
    "aile": "is_alle_column",
    "poids": "is_poids_column",
}

# Embedded fallback — keep in sync with config/column_schema.json.
DEFAULT_SCHEMA = {
    "columns": [
        {"role": "batch", "keywords": ["bague", "bague no", "numero", "nummer"], "width_share": [0.04, 0.14]},
        {"role": "species", "keywords": ["espece", "espèce", "art"], "width_share": [0.15, 0.38]},
        {"role": "sexe", "keywords": ["sexe", "d'sexe", "sex"], "width_share": [0.03, 0.11]},
        {"role": "age", "keywords": ["age", "âge", "alter"], "width_share": [0.05, 0.14]},
        {"role": "jour_mois", "keywords": ["jour mois", "jour", "mois"], "width_share": [0.05, 0.15]},
        {"role": "heure", "keywords": ["heure"], "width_share": [0.03, 0.11]},
        {"role": "aile", "keywords": ["aile", "alle", "aiie"], "width_share": [0.04, 0.13]},
        {"role": "poids", "keywords": ["poids", "polds"], "width_share": [0.03, 0.12]},
    ],
    "min_assign_score": 0.20,
    "gap_column_penalty": 0.05,
    "gap_role_penalty": 0.25,
}


def load_schema(path: Optional[Path] = None) -> dict:
    """User override first, then the canonical file, then embedded defaults."""
    candidates = (
        [Path(path)] if path else [CUSTOM_SCHEMA_PATH, DEFAULT_SCHEMA_PATH]
    )
    for target in candidates:
        if not target.exists():
            continue
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("columns"):
                return data
        except Exception as e:
            print(f"[column_schema] Konnte {target} nicht lesen ({e}) — überspringe.")
    return DEFAULT_SCHEMA


def save_schema(schema: dict, path: Optional[Path] = None) -> None:
    target = Path(path) if path else CUSTOM_SCHEMA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def reset_schema() -> bool:
    """Drop the user override; returns True when one existed."""
    if CUSTOM_SCHEMA_PATH.exists():
        CUSTOM_SCHEMA_PATH.unlink()
        return True
    return False


# ── UI bridge: schema <-> editable rows ────────────────────────────────
# The app shows the schema as a small editable table BEFORE processing.
# Rows carry: position, label (info), keywords, width range, active flag.


def schema_to_rows(schema: dict) -> List[dict]:
    rows = []
    for pos, col in enumerate(schema.get("columns", []), start=1):
        width = col.get("width_share", [0.0, 1.0])
        rows.append(
            {
                "Pos": pos,
                "Spalte": col.get("label", col.get("role", "?")),
                "Aktiv": bool(col.get("enabled", True)),
                "Header-Stichwörter": ", ".join(col.get("keywords", [])),
                "Breite min %": round(float(width[0]) * 100, 1),
                "Breite max %": round(float(width[1]) * 100, 1),
                "_role": col.get("role", ""),
            }
        )
    return rows


def rows_to_schema(rows: Sequence[dict], base_schema: Optional[dict] = None) -> dict:
    """Rebuild the schema from edited rows (sorted by Pos).

    Rows map back onto the base schema's roles via the hidden ``_role`` key —
    only order, keywords, width range and the active flag are user-editable.
    Invalid numbers fall back to the base values.
    """
    base = base_schema or load_schema()
    base_by_role = {c["role"]: c for c in base.get("columns", [])}

    columns = []
    for row in sorted(rows, key=lambda r: r.get("Pos") or 0):
        role = row.get("_role", "")
        base_col = base_by_role.get(role)
        if base_col is None:
            continue
        keywords = [
            k.strip()
            for k in str(row.get("Header-Stichwörter", "")).split(",")
            if k.strip()
        ] or list(base_col.get("keywords", []))
        try:
            lo = max(0.0, float(row.get("Breite min %", 0)) / 100.0)
            hi = min(1.0, float(row.get("Breite max %", 100)) / 100.0)
            if hi <= lo:
                raise ValueError
        except (TypeError, ValueError):
            lo, hi = base_col.get("width_share", [0.0, 1.0])

        new_col = dict(base_col)
        new_col["keywords"] = keywords
        new_col["width_share"] = [lo, hi]
        new_col["enabled"] = bool(row.get("Aktiv", True))
        columns.append(new_col)

    schema = dict(base)
    schema["columns"] = columns
    return schema


def _active_columns(schema: dict) -> List[dict]:
    return [c for c in schema.get("columns", []) if c.get("enabled", True)]


def keyword_score(header_text: str, keywords: Sequence[str]) -> float:
    """0..1 — how well the OCR'd header matches the role's keywords.

    Fuzzy, but word-based: the keyword must match the whole header or one of
    its words (OCR misreads like "Alle" for "Aile" or "d'Sexe ®" still hit).
    Deliberately NOT substring/partial matching — short fragments inside
    longer headers caused the old "'No' anywhere becomes the Bague column"
    bug and inflate cross-scores ("Jour Mois" must not look like "Poids").
    """
    text = fuzz_utils.default_process(header_text or "")
    if not text or len(text) < 2:
        return 0.0
    candidates = text.split() + ([text] if " " in text else [])
    best = 0.0
    for keyword in keywords:
        kw = fuzz_utils.default_process(keyword)
        if not kw:
            continue
        for cand in candidates:
            score = fuzz.ratio(kw, cand) / 100.0
            # Tiny header fragments ("No", "®") are junk unless they equal a
            # keyword exactly.
            if len(cand) <= 3 and cand != kw:
                score *= 0.2
            best = max(best, score)
    return best


def width_score(share: float, expected: Sequence[float]) -> float:
    """0..1 — 1 inside the expected [min, max] share, decaying outside."""
    if share <= 0:
        return 0.0
    lo, hi = float(expected[0]), float(expected[1])
    if lo <= share <= hi:
        return 1.0
    # Linear falloff over one full band width outside the range.
    band = max(hi - lo, 1e-6)
    distance = (lo - share) if share < lo else (share - hi)
    return max(0.0, 1.0 - distance / band)


def pair_score(header_text: str, share: float, role_def: dict) -> float:
    """Combined evidence that this extracted column is this role.

    Header keywords dominate (0.75) — they are the direct evidence; width
    (0.25) breaks ties and carries columns whose header OCR came back empty.
    """
    kw = keyword_score(header_text, role_def.get("keywords", []))
    ws = width_score(share, role_def.get("width_share", [0.0, 1.0]))
    return 0.75 * kw + 0.25 * ws


def assign_roles(
    header_texts: Sequence[str],
    widths: Sequence[float],
    schema: Optional[dict] = None,
) -> List[Optional[str]]:
    """Assign a role (or None) to every extracted column.

    Monotone alignment via dynamic programming: roles keep their printed
    order, each role is used at most once, columns can stay unassigned
    (extra grid slivers, the notes area) and roles can be skipped (column
    not extracted on this scan).
    """
    schema = schema or load_schema()
    roles = _active_columns(schema)
    min_score = float(schema.get("min_assign_score", 0.30))
    gap_col = float(schema.get("gap_column_penalty", 0.05))
    gap_role = float(schema.get("gap_role_penalty", 0.25))

    n = len(header_texts)
    m = len(roles)
    total_width = float(sum(widths)) or 1.0
    shares = [float(w) / total_width for w in widths]

    scores = [
        [pair_score(header_texts[i], shares[i], roles[j]) for j in range(m)]
        for i in range(n)
    ]

    # DP over (columns 0..n) x (roles 0..m); dp[i][j] = best total for the
    # first i columns and first j roles.
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_role
        move[0][j] = "skip_role"
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_col
        move[i][0] = "skip_col"

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            if j > 0:
                # Assign column i-1 to role j-1 — only with real evidence.
                s = scores[i - 1][j - 1]
                if s >= min_score and dp[i - 1][j - 1] > NEG:
                    cand = dp[i - 1][j - 1] + s
                    if cand > dp[i][j]:
                        dp[i][j] = cand
                        move[i][j] = "assign"
                # Skip role j-1 (not extracted / not assignable).
                if dp[i][j - 1] > NEG:
                    cand = dp[i][j - 1] - gap_role
                    if cand > dp[i][j]:
                        dp[i][j] = cand
                        move[i][j] = "skip_role"
            # Leave column i-1 unassigned.
            if dp[i - 1][j] > NEG:
                cand = dp[i - 1][j] - gap_col
                if cand > dp[i][j]:
                    dp[i][j] = cand
                    move[i][j] = "skip_col"

    # Traceback.
    assignment: List[Optional[str]] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        action = move[i][j]
        if action == "assign":
            assignment[i - 1] = roles[j - 1]["role"]
            i, j = i - 1, j - 1
        elif action == "skip_role":
            j -= 1
        else:  # skip_col (also covers the i>0, j==0 edge)
            i -= 1
    return assignment


def flags_for_role(role: Optional[str]) -> Dict[str, bool]:
    """Complete is_*-flag dict for a role (all False for None/unknown)."""
    return {flag: (ROLE_TO_FLAG.get(role) == flag) for flag in ROLE_TO_FLAG.values()}
