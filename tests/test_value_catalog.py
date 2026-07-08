"""Tests for the editable option catalogs (libs/value_catalog.py)."""

from libs.value_catalog import (
    DEFAULTS,
    add_option,
    load_options,
    options_with_values,
    remove_option,
)


def test_defaults_cover_bander_notation():
    sexe = DEFAULTS["sexe"]
    for v in ("", "m", "w", "f", "(m)", "(f)", "?", "X", '"'):
        assert v in sexe, f"{v!r} fehlt in den Sexe-Defaults"


def test_load_falls_back_to_defaults(tmp_path):
    assert load_options("sexe", path=tmp_path / "nope.json") == DEFAULTS["sexe"]


def test_add_and_reload(tmp_path):
    p = tmp_path / "sexe_options.json"
    ok, msg = add_option("sexe", "(m)?", path=p)
    assert ok
    options = load_options("sexe", path=p)
    assert "(m)?" in options
    assert "" in options  # empty entry always present

    ok, msg = add_option("sexe", "(m)?", path=p)
    assert not ok  # duplicate


def test_remove_option(tmp_path):
    p = tmp_path / "sexe_options.json"
    add_option("sexe", "zzz", path=p)
    ok, _ = remove_option("sexe", "zzz", path=p)
    assert ok
    assert "zzz" not in load_options("sexe", path=p)

    ok, msg = remove_option("sexe", "", path=p)
    assert not ok


def test_options_with_values_includes_data_values(tmp_path):
    """Values already present in cells must be selectable even if they are
    not in the catalog — a Selectbox renders unknown values as missing."""
    p = tmp_path / "sexe_options.json"
    options = options_with_values("sexe", ["m", "Sonderwert", "", '"'], path=p)
    assert "Sonderwert" in options
    assert options.count("m") == 1
