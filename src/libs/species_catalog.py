"""Helpers to read, extend, and persist the bird-species class list.

The canonical store is `config/class_indices.json` (the file the
`FuzzyMatchingBirdNames` stage reads). It's a flat ``{name: index}`` dict —
indices are unique but not otherwise meaningful for our use, they exist
because the format originated from a Keras model's `class_indices`.

We add a sibling ``config/class_indices_custom.json`` so user-added names
survive symlinks / read-only deployments without ever modifying the
upstream file. Both are merged at load time.
"""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_CATALOG = Path("./config/class_indices.json")
CUSTOM_CATALOG = Path("./config/class_indices_custom.json")


def _normalise(name: str) -> str:
    """Compare names case-insensitively and ignoring diacritics so we don't
    add 'Buchfink' twice if it's already there as 'buchfink' or 'Bûchfink'."""
    n = unicodedata.normalize("NFKD", (name or "").strip())
    n = "".join(c for c in n if not unicodedata.combining(c))
    return n.casefold()


def _load_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    # Accept list form too — older callers store the class list as a JSON array.
    if isinstance(data, list):
        return {str(name): idx for idx, name in enumerate(data)}
    return {}


def load_catalog(
    path: Optional[Path] = None,
    custom_path: Optional[Path] = None,
) -> List[str]:
    """Return the sorted union of canonical + custom species names.

    Names are NFC-normalised: the historical class_indices.json stores
    umlauts decomposed (o + combining diaeresis), which made recognised
    values compare unequal to visually identical NFC text ("Hausrötel" !=
    "Hausrötel") in exports and downstream tools.
    """
    default = _load_file(path or DEFAULT_CATALOG)
    custom = _load_file(custom_path or CUSTOM_CATALOG)
    merged = {**default, **custom}
    # Skip the historical "ditto mark" entry — it isn't a species.
    names = {
        unicodedata.normalize("NFC", k)
        for k in merged.keys()
        if k.strip() and k != '"'
    }
    return sorted(names)


def is_in_catalog(name: str, catalog: Optional[Iterable[str]] = None) -> bool:
    if not name or not name.strip():
        return False
    if catalog is None:
        catalog = load_catalog()
    target = _normalise(name)
    return any(_normalise(c) == target for c in catalog)


def add_to_catalog(name: str, custom_path: Optional[Path] = None) -> tuple[bool, str]:
    """Add ``name`` to the user-extensions file. Returns ``(added, message)``.

    - ``added=True`` if the name was new and written.
    - ``added=False`` if it already existed (canonical or custom) — message
      explains why.
    """
    name = (name or "").strip()
    if not name:
        return False, "Name ist leer."
    if name == '"':
        return False, "Anführungszeichen sind reserviert für Ditto-Marker."

    if is_in_catalog(name):
        return False, f"„{name}“ steht bereits im Katalog."

    target_path = custom_path or CUSTOM_CATALOG
    custom = _load_file(target_path)
    # Pick an index that doesn't collide with the default file.
    default = _load_file(DEFAULT_CATALOG)
    used = set(default.values()) | set(custom.values())
    next_idx = max(used, default=-1) + 1
    custom[name] = next_idx

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(custom, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return True, f"„{name}“ hinzugefügt."
