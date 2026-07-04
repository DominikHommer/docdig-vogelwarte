"""Tests for the user-extensible bird-species catalog."""

import json
from pathlib import Path

import pytest

from libs import species_catalog


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    """Point the species catalog at a clean tmpdir."""
    default = tmp_path / "class_indices.json"
    custom = tmp_path / "class_indices_custom.json"
    default.write_text(json.dumps({"Buchfink": 1, "Bergfink": 2}))
    monkeypatch.setattr(species_catalog, "DEFAULT_CATALOG", default)
    monkeypatch.setattr(species_catalog, "CUSTOM_CATALOG", custom)
    return default, custom


def test_load_merges_canonical_and_custom(tmp_catalog):
    default, custom = tmp_catalog
    custom.write_text(json.dumps({"Erlenzeisig": 100}))
    names = species_catalog.load_catalog()
    assert names == ["Bergfink", "Buchfink", "Erlenzeisig"]


def test_load_filters_ditto_placeholder(tmp_catalog):
    default, _ = tmp_catalog
    default.write_text(json.dumps({'"': 0, "Buchfink": 1}))
    assert species_catalog.load_catalog() == ["Buchfink"]


def test_is_in_catalog_is_diacritic_insensitive(tmp_catalog):
    default, _ = tmp_catalog
    default.write_text(json.dumps({"Hänfling": 1}))
    assert species_catalog.is_in_catalog("Hänfling")
    assert species_catalog.is_in_catalog("hänfling")
    assert species_catalog.is_in_catalog("Hanfling")  # diacritic stripped
    assert not species_catalog.is_in_catalog("Buchfink")


def test_add_new_name_writes_to_custom(tmp_catalog):
    default, custom = tmp_catalog
    ok, msg = species_catalog.add_to_catalog("Heckenbraunelle")
    assert ok, msg
    assert custom.exists()
    saved = json.loads(custom.read_text())
    assert "Heckenbraunelle" in saved


def test_add_skips_duplicates(tmp_catalog):
    species_catalog.add_to_catalog("Tannenmeise")
    ok, msg = species_catalog.add_to_catalog("Tannenmeise")
    assert not ok
    assert "bereits" in msg.lower()


def test_add_rejects_ditto_placeholder(tmp_catalog):
    ok, msg = species_catalog.add_to_catalog('"')
    assert not ok
    assert "anführungs" in msg.lower() or "ditto" in msg.lower() or "reserv" in msg.lower()


def test_add_rejects_empty(tmp_catalog):
    ok, msg = species_catalog.add_to_catalog("   ")
    assert not ok


def test_custom_entries_get_unique_indices(tmp_catalog):
    species_catalog.add_to_catalog("A")
    species_catalog.add_to_catalog("B")
    custom = json.loads((tmp_catalog[1]).read_text())
    assert custom["A"] != custom["B"]
    # No collision with canonical indices either
    default = json.loads(tmp_catalog[0].read_text())
    assert set(custom.values()).isdisjoint(default.values())


def test_load_after_add_includes_new_name(tmp_catalog):
    species_catalog.add_to_catalog("Wendehals")
    names = species_catalog.load_catalog()
    assert "Wendehals" in names
