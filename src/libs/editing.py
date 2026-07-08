"""Apply editor cell changes to the prediction structure — no Streamlit.

The Streamlit app feeds `st.data_editor`'s ``edited_rows`` deltas through
:func:`apply_cell_edit`. Keeping the mutation logic here makes the exact
editing semantics (manual-edit protection, bague anchor rebuild, unknown
species tracking, ditto marks) unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from libs.bague_sequence import mark_manual_edit, rebuild_batch_sequence
from libs.species_catalog import is_in_catalog


@dataclass
class EditResult:
    changed: bool = False
    anchor_rebuilt: bool = False
    rebuild_stats: Optional[dict] = None
    unknown_species: Optional[str] = None


def apply_cell_edit(
    page: dict,
    col_i: int,
    row_i: int,
    new_value,
    species_catalog: Optional[Sequence[str]] = None,
) -> EditResult:
    """Write one user edit into ``page["columns"][col_i]["cells"][row_i]``.

    ``row_i`` is the CELL index (1 = first data row — index 0 is the header).

    Semantics:
    - identical value -> no-op
    - bague anchor cell -> rebuild the whole column from the new anchor
      (letter prefixes like "A90401" survive, see bague_sequence)
    - any other cell -> value is kept verbatim (including '"' ditto marks),
      marked as a manual edit so no recognizer or rebuild overwrites it
    - unknown species names are reported so the UI can offer to save them
    """
    result = EditResult()
    columns = page.get("columns", [])
    if col_i >= len(columns):
        return result
    column = columns[col_i]
    cells = column.get("cells", [])
    if not (1 <= row_i < len(cells)):
        return result

    cell = cells[row_i]
    new_value = "" if new_value is None else str(new_value).strip()
    old_value = (cell.get("erkannt") or "").strip()
    if new_value == old_value:
        return result

    result.changed = True
    cell["erkannt"] = new_value

    if column.get("is_batch_column", False) and cell.get("is_anchor"):
        result.anchor_rebuilt = True
        result.rebuild_stats = rebuild_batch_sequence(column, anchor_value=new_value)
    else:
        mark_manual_edit(cell)
        cell["score"] = 100

    if (
        column.get("is_species_column", False)
        and new_value
        and new_value != '"'
        and not is_in_catalog(new_value, catalog=species_catalog)
    ):
        result.unknown_species = new_value

    return result


def apply_editor_deltas(
    page: dict,
    edited_rows: dict,
    displayed_index: Sequence[int],
    visible: Sequence[Tuple[int, str]],
    species_catalog: Optional[Sequence[str]] = None,
) -> List[EditResult]:
    """Apply a `st.data_editor` ``edited_rows`` delta dict.

    Args:
        page: the prediction page to mutate.
        edited_rows: ``{display_position: {column_label: value, ...}, ...}``
            straight from ``st.session_state[editor_key]["edited_rows"]``.
        displayed_index: the DataFrame index of the *displayed* frame — maps
            display position -> row number (1-based cell index). With the
            review filter active this is a subset of all rows.
        visible: [(col_idx, label), ...] mapping from page_to_dataframe.

    Idempotent: reapplying the same delta is a no-op, so accumulating
    ``edited_rows`` state across reruns is safe.
    """
    label_to_col = {label: col_i for col_i, label in visible}
    results: List[EditResult] = []

    for position, changes in sorted(edited_rows.items()):
        position = int(position)
        if position < 0 or position >= len(displayed_index):
            continue
        row_i = int(displayed_index[position])
        for label, value in changes.items():
            col_i = label_to_col.get(label)
            if col_i is None:
                continue  # helper columns (✓, ⚠, 📷 ...) are read-only
            results.append(
                apply_cell_edit(
                    page, col_i, row_i, value, species_catalog=species_catalog
                )
            )
    return results
