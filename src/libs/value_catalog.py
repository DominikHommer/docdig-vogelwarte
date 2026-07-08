"""Small JSON-backed option catalogs for closed-vocabulary editor columns.

The Sexe column (and potentially others) offers a dropdown in the editor.
Its options must be user-extendable — banders use variants like ``(m)?`` —
so the list lives in ``config/<name>_options.json`` and can be managed from
the app's sidebar, exactly like the species catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


CONFIG_DIR = Path("./config")

# Defaults per catalog. "" = empty cell; '"' = ditto mark (must be listed,
# otherwise the Selectbox renders ditto cells as missing values).
DEFAULTS = {
    "sexe": ["", "m", "w", "f", "(m)", "(w)", "(f)", "m?", "w?", "f?", "?", "X", '"'],
}


def _path(name: str, path: Optional[Path] = None) -> Path:
    return path or (CONFIG_DIR / f"{name}_options.json")


def load_options(name: str, path: Optional[Path] = None) -> List[str]:
    """Options for ``name`` — user file if present, else the defaults."""
    target = _path(name, path)
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, list):
                options = [str(x) for x in data]
                # "" must always be available so cells can be emptied.
                if "" not in options:
                    options.insert(0, "")
                return options
        except Exception:
            pass
    return list(DEFAULTS.get(name, [""]))


def add_option(name: str, value: str, path: Optional[Path] = None) -> Tuple[bool, str]:
    """Append ``value`` to the catalog. Returns (added, message)."""
    value = (value or "").strip()
    if not value:
        return False, "Wert ist leer."

    options = load_options(name, path)
    if value in options:
        return False, f"„{value}“ steht bereits in der Liste."

    options.append(value)
    target = _path(name, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True, f"„{value}“ hinzugefügt."


def remove_option(name: str, value: str, path: Optional[Path] = None) -> Tuple[bool, str]:
    """Remove ``value`` (except the empty entry). Returns (removed, message)."""
    if value == "":
        return False, "Der leere Eintrag kann nicht entfernt werden."
    options = load_options(name, path)
    if value not in options:
        return False, f"„{value}“ steht nicht in der Liste."
    options = [o for o in options if o != value]
    target = _path(name, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True, f"„{value}“ entfernt."


def options_with_values(name: str, present_values: Sequence[str], path: Optional[Path] = None) -> List[str]:
    """Catalog options plus whatever already occurs in the data.

    A Selectbox renders values outside its option list as missing — so any
    value that is already in a cell must be selectable, whether or not it is
    in the catalog.
    """
    options = load_options(name, path)
    for value in present_values:
        value = (value or "").strip()
        if value and value not in options:
            options.append(value)
    return options
