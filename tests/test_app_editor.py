"""AppTest: the editor view renders with the fragment architecture.

Simulates a processed session (synthetic predictions + fixture scan image)
and drives the app through the editor path. The edit semantics themselves
are unit-tested in test_editing.py — here we assert the page builds, the
cache behaves, and a rerun keeps the editor stable.
"""

import numpy as np
import pytest

from .conftest import project_root

pytestmark = pytest.mark.skipif(
    not (project_root() / "src" / "app.py").exists(), reason="app missing"
)


def _cell(erkannt="", score=90, **extra):
    img = np.full((40, 100), 255, dtype=np.uint8)
    img[10:30, 20:80] = 0
    cell = {
        "image": img,
        "image_raw": img,
        "erkannt": erkannt,
        "score": score,
        "skip_ocr": False,
        "verbesserung": "",
    }
    cell.update(extra)
    return cell


def _column(flag, values, **col_extra):
    cells = [_cell()] + [_cell(v) for v in values]
    column = {
        "cells": cells,
        "is_batch_column": False,
        "is_species_column": False,
        "is_sexe_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }
    column[flag] = True
    column.update(col_extra)
    return column


def _fake_page():
    bague = _column("is_batch_column", ["90401", "90402", "90403"])
    bague["cells"][1]["is_anchor"] = True
    return {
        "columns": [
            bague,
            _column("is_species_column", ["Buchfink", '"', "Erlenzeisig"]),
            _column("is_sexe_column", ["m", "", "w"]),
            _column("is_alle_column", ["68", "72", "75"]),
        ]
    }


def _scan_path() -> str:
    fixture = project_root() / "tests" / "fixtures" / "cells" / "age"
    pngs = sorted(fixture.glob("*.png"))
    if pngs:
        return str(pngs[0])
    pytest.skip("no fixture image available as fake scan")


def _editor_apptest():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/app.py", default_timeout=120)
    scan = _scan_path()
    at.session_state["uploaded"] = True
    at.session_state["processed"] = True
    at.session_state["uploaded_name"] = "test.pdf"
    at.session_state["pdf_path"] = scan  # unused in editor view
    at.session_state["all_pages"] = [scan]
    at.session_state["page_static_urls"] = [None]
    at.session_state["predictions"] = [_fake_page()]
    at.session_state["page_idx"] = 0
    return at


def test_editor_view_renders_without_exception(monkeypatch):
    monkeypatch.chdir(project_root())
    at = _editor_apptest()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # Page view cache was built exactly once and carries the display index.
    cache = at.session_state["editor_cache"]
    assert 0 in cache
    entry = cache[0]
    assert entry["labels"][0] == "Bague"
    assert entry["display_index"] == [1, 2, 3]
    assert len(entry["review_items"]) == 0


def test_editor_second_run_reuses_cache(monkeypatch):
    """A rerun must NOT rebuild the page view (that is what keeps the grid
    scroll/focus stable) — the cached entry stays the same object."""
    monkeypatch.chdir(project_root())
    at = _editor_apptest()
    at.run()
    entry_before = at.session_state["editor_cache"][0]
    df_id = id(entry_before["df_status"])
    at.run()
    assert not at.exception
    entry_after = at.session_state["editor_cache"][0]
    assert id(entry_after["df_status"]) == df_id, "page view was rebuilt on rerun"


def test_editor_csv_download_has_no_thumbnails(monkeypatch):
    monkeypatch.chdir(project_root())
    at = _editor_apptest()
    at.run()
    from libs.table_view import build_csv_bytes

    csv_text = build_csv_bytes(at.session_state["predictions"]).decode("utf-8-sig")
    assert "data:image" not in csv_text
    assert "Buchfink" in csv_text
