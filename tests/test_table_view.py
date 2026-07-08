"""Tests for the pure table-view layer (libs/table_view.py)."""

import numpy as np

from libs.table_view import (
    build_csv_bytes,
    cell_image_data_uri,
    compute_row_confidence,
    page_to_dataframe,
    visible_columns,
)


def _img():
    img = np.full((40, 100), 255, dtype=np.uint8)
    img[10:30, 20:80] = 0
    return img


def _cell(erkannt="", score=-1, **extra):
    cell = {
        "image": _img(),
        "image_raw": _img(),
        "erkannt": erkannt,
        "score": score,
        "skip_ocr": False,
        "verbesserung": "",
    }
    cell.update(extra)
    return cell


def _page():
    def col(flag, values, scores=None):
        scores = scores or [90] * len(values)
        c = {
            "cells": [_cell()] + [_cell(v, s) for v, s in zip(values, scores)],
            "is_batch_column": False,
            "is_species_column": False,
            "is_sexe_column": False,
            "is_age_column": False,
            "is_jour-mois_column": False,
            "is_heure_column": False,
            "is_alle_column": False,
            "is_poids_column": False,
        }
        c[flag] = True
        return c

    return {
        "columns": [
            col("is_batch_column", ["90401", "90402"]),
            col("is_species_column", ["Buchfink", '"']),
            col("is_alle_column", ["68", "72"]),
        ]
    }


def test_csv_export_contains_no_image_data():
    """Jan's bug: the Nextcloud CSV had a data:image... column per value
    column. The export must never contain thumbnails."""
    csv_bytes = build_csv_bytes([_page()])
    text = csv_bytes.decode("utf-8-sig")
    assert "data:image" not in text
    assert "📷" not in text
    assert "Buchfink" in text
    assert "90401" in text


def test_csv_header_matches_visible_labels():
    text = build_csv_bytes([_page()]).decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0] == "Seite 1"
    assert lines[1] == "Bague;Espèce;Aile"


def test_editor_dataframe_has_thumbnails_csv_frame_does_not():
    page = _page()
    df_editor, _ = page_to_dataframe(page, include_thumbnails=True)
    df_csv, _ = page_to_dataframe(page, include_thumbnails=False)
    assert any(c.startswith("📷") for c in df_editor.columns)
    assert not any(c.startswith("📷") for c in df_csv.columns)
    assert df_editor["Bague"].tolist() == df_csv["Bague"].tolist()


def test_thumbnail_uri_roundtrip():
    uri = cell_image_data_uri(_img())
    assert uri.startswith("data:image/png;base64,")
    assert cell_image_data_uri(None) == ""


def test_row_confidence_thresholds():
    page = _page()
    # row 1: scores 90/90/90 -> 🟡 ; row 2: force one low score -> 🔴
    page["columns"][2]["cells"][2]["score"] = 20
    visible = visible_columns(page)
    conf = compute_row_confidence(page, visible, num_rows=2)
    assert conf[0][0] == "🟡"
    assert conf[1][0] == "🔴"


def test_visible_columns_orders_and_disambiguates():
    page = _page()
    page["columns"].append(dict(page["columns"][1]))  # second species column
    labels = [label for _, label in visible_columns(page)]
    assert labels[0] == "Bague"
    assert "Espèce" in labels and "Espèce (2)" in labels
