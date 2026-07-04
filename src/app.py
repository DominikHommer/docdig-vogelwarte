"""Streamlit UI for DocDig — bird-banding list digitisation.

Layout:
- Sidebar: file info, page navigation, stats, CSV export.
- Main:    original scan on the left, editable table on the right.
- Editing:
    * Bague anchor cell -> resyncs the whole column (via rebuild_batch_sequence)
    * Any other cell    -> marked as manual edit, kept verbatim from then on
- The CellDenoiser was dropped from the pipeline (made downstream OCR worse).
  Cells stay dark text on white background end to end.
"""

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer

load_dotenv()  # .env: Nextcloud-Zugang etc. (see libs/nextcloud.py)

from pipeline.cv_pipeline import CVPipeline
from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.merged_column_extractor import MergedColumnExtractor
from modules.merged_row_extractor import MergedRowExtractor
from modules.detect_columns import DetectColumns
from modules.cell_formatter import CellFormatter
from modules.quotation_mark_detector import QuotationMarkDetector
from modules.htr_vt_recognizer import HtrVtRecognizer
from modules.digit_recognizer import DigitRecognizer
from modules.sexe_classifier import SexeClassifier
from modules.trocr import TrOCR
from modules.fuzzy_matching import FuzzyMatchingBirdNames, FuzzyMatchingAge
from modules.numeric_consensus import NumericConsensus
from libs.bague_sequence import mark_manual_edit, rebuild_batch_sequence
from libs.columns import COLUMN_FLAG_TO_LABEL, column_label as _column_label
from libs.nextcloud import (
    DEFAULT_REMOTE_DIR as NEXTCLOUD_DEFAULT_DIR,
    NextcloudConfig,
    enable_system_truststore,
    load_config as nextcloud_env_config,
    probe as nextcloud_probe,
    upload_results as nextcloud_upload_results,
)

# Intern signierte Zertifikate (nextcloud.vogelwarte.ch) über den
# Betriebssystem-Truststore verifizieren statt über Pythons certifi-Liste.
enable_system_truststore()
from libs.species_catalog import (
    add_to_catalog as species_add,
    is_in_catalog as species_known,
    load_catalog as species_load,
)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# Every extracted column is shown — the semantic ones (COLUMN_FLAG_TO_LABEL,
# see libs/columns.py) with their form label, everything else as "Spalte N".

# Order the columns in the editor the way they appear on the forms.
VISIBLE_COLUMN_ORDER = list(COLUMN_FLAG_TO_LABEL.values())

# m/w kommen vom deutschsprachigen Beringer (M-W-Classification), f/(f) aus
# den französischen Formularen.
SEXE_OPTIONS = ["", "m", "w", "f", "(f)", "?", "X"]

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")


# ──────────────────────────────────────────────────────────────────────
# Page config + global styles
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocDig", page_icon="🦜", layout="wide")

st.markdown(
    """
    <style>
    /* Tighter row heights in the data editor */
    [data-testid="stDataEditor"] tbody tr td { padding: 2px 6px; }

    /* Slightly tighter top padding — more space for the table */
    .block-container { padding-top: 1.5rem; }

    /* Scan column: deliberately NO position:sticky and NO z-index overrides
       — position:sticky always creates a stacking context, which traps
       Streamlit's fullscreen overlay inside the column and makes the
       exit/collapse button unreachable (fullscreen could not be closed).
       The scan preview lives in st.container(height=...) instead: a plain
       scroll container that neither traps position:fixed nor stacking. */

    /* Confidence pills */
    .pill-ok { background: #d4edda; color: #155724; }
    .pill-warn { background: #fff3cd; color: #856404; }
    .pill-bad { background: #f8d7da; color: #721c24; }
    .pill {
        padding: 1px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# Pipeline construction (lazy-loaded on first run)
# ──────────────────────────────────────────────────────────────────────
def build_page_pipeline(page_paths):
    pipeline = CVPipeline(input_data={"pdf-converter": page_paths})
    pipeline.add_stage(TableRotator(debug=False))
    pipeline.add_stage(TatrExtractor(debug=False))
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    pipeline.add_stage(DetectColumns())
    pipeline.add_stage(CellFormatter())
    pipeline.add_stage(QuotationMarkDetector())
    # Specialised recognisers per column type.
    pipeline.add_stage(HtrVtRecognizer())   # Espèce + Age (Fd/Fnd)
    # DOCDIG_DIGIT_BACKENDS steuert die Ziffern-Stimmen, z. B.
    #   DOCDIG_DIGIT_BACKENDS=yolo streamlit run src/app.py   (nur YOLO)
    #   DOCDIG_DIGIT_BACKENDS=tesseract,keras                 (ohne YOLO)
    _digit_backends = os.environ.get("DOCDIG_DIGIT_BACKENDS", "").strip()
    pipeline.add_stage(
        DigitRecognizer(
            backends=tuple(_digit_backends.split(",")) if _digit_backends else None
        )
    )   # Bague (seq) + alle Zahlen-Spalten (CRNN/Tesseract + YOLO)
    pipeline.add_stage(SexeClassifier())    # Sexe (or no-op without model)
    pipeline.add_stage(TrOCR())             # Second opinion für Espèce + Zahlen-Spalten
    pipeline.add_stage(FuzzyMatchingBirdNames())  # dual-source consensus for Espèce
    pipeline.add_stage(NumericConsensus())  # 3-Stimmen-Voting für Aile/Poids/Heure/Jour-Mois
    pipeline.add_stage(FuzzyMatchingAge())
    return pipeline


def build_base_pipeline():
    """Just the PDF → page-jpg conversion."""
    p = CVPipeline()
    p.add_stage(PdfConverter(debug=False))
    return p


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def visible_columns(page: dict):
    """Return [(col_idx_in_predictions, label), ...] in the order to display.

    Every extracted column is included: semantically tagged columns get their
    form label, untagged ones show up as "Spalte N" at their scan position.
    """
    labelled = []
    seen: dict[str, int] = {}
    for i, col in enumerate(page.get("columns", [])):
        label = _column_label(col, index=i)
        # Detector can tag two columns identically — disambiguate so the
        # DataFrame keeps one column per extracted column.
        if label in seen:
            seen[label] += 1
            label = f"{label} ({seen[label]})"
        else:
            seen[label] = 1
        labelled.append((i, label))
    # Sort to follow VISIBLE_COLUMN_ORDER; unknown labels keep scan order at the end.
    order = {name: rank for rank, name in enumerate(VISIBLE_COLUMN_ORDER)}
    labelled.sort(key=lambda pair: (order.get(pair[1], 999), pair[0]))
    return labelled


def _cell_image_data_uri(image, max_h: int = 36, max_w: int = 220) -> str:
    """Encode a cell crop as a base64 data URI for `st.column_config.ImageColumn`.

    Resizes to a thumbnail so the editor stays snappy. Returns an empty string
    when the crop is missing or unreadable.
    """
    if image is None:
        return ""
    try:
        import cv2

        arr = np.asarray(image)
        if arr.size == 0:
            return ""
        if arr.dtype != np.uint8:
            mx = float(arr.max()) if arr.size else 1.0
            if 0.0 <= float(arr.min()) and mx <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[:, :, :3]

        h, w = arr.shape[:2]
        scale = min(max_h / max(h, 1), max_w / max(w, 1), 1.0)
        if scale < 1.0:
            arr = cv2.resize(
                arr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            return ""
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return ""


def page_to_dataframe(page: dict, include_thumbnails: bool = True):
    """Build (df, [(col_idx, label), ...]) for one page. Every extracted
    column is included (semantic label or "Spalte N" fallback).

    With ``include_thumbnails=True`` each value column is preceded by a `📷`
    column containing the base64-encoded cell crop, so the user can verify
    what the OCR actually saw without leaving the table.
    """
    columns = page.get("columns", [])
    visible = visible_columns(page)
    if not visible:
        return pd.DataFrame(), []

    max_data_rows = max((len(col.get("cells", [])) - 1) for col in columns) if columns else 0
    max_data_rows = max(max_data_rows, 0)

    ordered_keys: list[str] = []
    data: dict = {}

    for col_i, label in visible:
        img_key = f"📷 {label}" if include_thumbnails else None
        if img_key:
            ordered_keys.append(img_key)
            data[img_key] = []
        ordered_keys.append(label)
        data[label] = []

    for row_i in range(1, max_data_rows + 1):
        for col_i, label in visible:
            cells = columns[col_i].get("cells", [])
            cell = cells[row_i] if row_i < len(cells) else None
            value = (cell.get("erkannt") if cell else "") or ""
            data[label].append(value.strip())
            if include_thumbnails:
                img_key = f"📷 {label}"
                source = cell.get("image_raw") if cell else None
                if source is None and cell is not None:
                    source = cell.get("image")
                data[img_key].append(_cell_image_data_uri(source))

    df = pd.DataFrame(data, columns=ordered_keys)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name="#")
    return df, visible


def sync_edits(page_idx: int, edited_df: pd.DataFrame, visible) -> int:
    """Push user edits in the data editor back into st.session_state.predictions.

    `visible` is the [(col_idx, label), ...] mapping from `page_to_dataframe`.
    Returns the number of cells that were updated (for the toast).

    Side effect: if the user typed an Espèce that is not in the species
    catalog, the name gets queued in ``st.session_state.pending_species`` so
    the UI can offer to save it permanently.
    """
    pred = st.session_state.predictions[page_idx]
    columns = pred["columns"]
    changed = 0
    bague_rebuilds = []
    pending = st.session_state.setdefault("pending_species", [])
    known_catalog = species_load()

    for col_i, label in visible:
        if col_i >= len(columns) or label not in edited_df.columns:
            continue
        col = columns[col_i]
        cells = col.get("cells", [])
        is_bague = col.get("is_batch_column", False)
        is_species = col.get("is_species_column", False)

        for row_offset, new_value in enumerate(edited_df[label].fillna("").tolist()):
            cell_i = row_offset + 1  # skip header
            if cell_i >= len(cells):
                break
            cell = cells[cell_i]
            old_value = (cell.get("erkannt") or "").strip()
            new_value = (new_value or "").strip()
            if new_value == old_value:
                continue

            cell["erkannt"] = new_value
            changed += 1

            if is_bague and cell.get("is_anchor"):
                bague_rebuilds.append((col, new_value))
            elif is_bague:
                mark_manual_edit(cell)
                cell["score"] = 100
            else:
                cell["skip_ocr"] = True
                cell["score"] = 100

            # Track unknown species so the UI can offer to persist them.
            if is_species and new_value and new_value != '"':
                if not species_known(new_value, catalog=known_catalog):
                    key = (page_idx, col_i, cell_i, new_value)
                    if key not in pending:
                        pending.append(key)

    for col, new_value in bague_rebuilds:
        stats = rebuild_batch_sequence(col, anchor_value=new_value)
        st.toast(
            f"🔄 Bague-Spalte aus Anker {stats['anchor']} neu berechnet "
            f"({stats['filled']} Zellen)."
        )

    return changed


def build_csv() -> bytes:
    """Build a single CSV from all pages. Same layout as the editor table."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    for page_idx, page in enumerate(st.session_state.predictions):
        if page is None:
            continue
        df, visible = page_to_dataframe(page)
        writer.writerow([f"Seite {page_idx + 1}"])
        writer.writerow([label for _, label in visible])
        for _, row in df.iterrows():
            writer.writerow(row.tolist())
        writer.writerow([])
    return output.getvalue().encode("utf-8-sig")


def reset_session():
    # Nextcloud-Login übersteht "Neue Datei" — der Rest wird verworfen.
    keep = {"nextcloud_config"}
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]


def get_nextcloud_config() -> NextcloudConfig | None:
    """Active Nextcloud login of THIS browser session.

    Credentials live only in st.session_state — never on disk, never shared
    between users of a deployed instance. A configured .env merely pre-fills
    the login form (see the sidebar expander).
    """
    return st.session_state.get("nextcloud_config")


def render_nextcloud_login():
    """Sidebar expander: Nextcloud-Login mit Verbindungstest."""
    config = get_nextcloud_config()
    env_defaults = nextcloud_env_config()

    with st.expander(
        "☁️ Nextcloud" + (" ✓" if config else ""), expanded=False
    ):
        if config:
            st.caption(f"Verbunden als **{config.user}** → `{config.remote_dir}`")
            if st.button("Abmelden", use_container_width=True, key="nc-logout"):
                del st.session_state["nextcloud_config"]
                st.rerun()
            return

        st.caption(
            "Zugang gilt nur für diese Browser-Sitzung und wird nirgends "
            "gespeichert. App-Passwort: Nextcloud → Einstellungen → "
            "Sicherheit → „Neues App-Passwort erstellen“."
        )
        with st.form("nextcloud-login", border=False):
            url = st.text_input(
                "Server",
                value=(env_defaults.base_url if env_defaults else "https://nextcloud.vogelwarte.ch"),
            )
            user = st.text_input(
                "Benutzername",
                value=(env_defaults.user if env_defaults else ""),
            )
            password = st.text_input("App-Passwort", type="password")
            remote_dir = st.text_input(
                "Ordner",
                value=(env_defaults.remote_dir if env_defaults else NEXTCLOUD_DEFAULT_DIR),
            )
            skip_ssl = st.checkbox(
                "Zertifikatsprüfung deaktivieren",
                value=(env_defaults.verify is False if env_defaults else False),
                help="Nur nötig, wenn der Server ein selbst-/intern signiertes "
                "Zertifikat nutzt und die Verbindung sonst mit einem "
                "SSL-Fehler scheitert. Deaktiviert die Echtheitsprüfung des "
                "Servers — nur für die interne Vogelwarte-Instanz verwenden.",
            )
            submitted = st.form_submit_button(
                "Verbinden", use_container_width=True, type="primary"
            )

        if submitted:
            if not (url.strip() and user.strip() and password.strip()):
                st.warning("Server, Benutzername und App-Passwort ausfüllen.")
                return
            candidate = NextcloudConfig(
                base_url=url.strip(),
                user=user.strip(),
                password=password.strip(),
                remote_dir=("/" + remote_dir.strip().lstrip("/")) if remote_dir.strip() else NEXTCLOUD_DEFAULT_DIR,
                verify=False if skip_ssl else (env_defaults.verify if env_defaults else True),
            )
            with st.spinner("Prüfe Verbindung …"):
                ok, message = nextcloud_probe(candidate)
            if ok:
                st.session_state["nextcloud_config"] = candidate
                st.toast(f"☁️ {message}")
                st.rerun()
            else:
                st.error(message)


# ──────────────────────────────────────────────────────────────────────
# Sidebar — compact: file info, CSV download, help.
# (Page navigation lives in the main header now.)
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦜 DocDig")
    st.caption("Vogelwarte – Beringungslisten")

    if st.session_state.get("uploaded_name"):
        st.divider()
        st.markdown(f"📄 **{st.session_state.uploaded_name}**")

        if st.session_state.get("predictions") and any(
            p is not None for p in st.session_state.predictions
        ):
            csv_bytes = build_csv()
            csv_name = f"{Path(st.session_state.uploaded_name).stem}.csv"
            st.download_button(
                "📥 CSV herunterladen",
                csv_bytes,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

            # Nextcloud: PDF + CSV land side by side under /docdig/<scan>/.
            # Login happens in the sidebar expander (session-only).
            nc_config = get_nextcloud_config()
            if nc_config:
                if st.button("☁️ In Nextcloud speichern", use_container_width=True):
                    try:
                        with st.spinner("Lade PDF + CSV nach Nextcloud hoch …"):
                            remote_pdf, remote_csv = nextcloud_upload_results(
                                pdf_path=st.session_state.pdf_path,
                                csv_bytes=csv_bytes,
                                csv_name=csv_name,
                                config=nc_config,
                            )
                        st.success(f"Hochgeladen: {remote_pdf} + {remote_csv}")
                    except Exception as e:
                        st.error(f"Nextcloud-Upload fehlgeschlagen: {e}")
            else:
                st.caption(
                    "☁️ Für den Nextcloud-Upload unten im Abschnitt "
                    "**Nextcloud** anmelden."
                )
        if st.button("🔄 Neue Datei", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()
        # ── Vogelarten-Katalog verwalten ───────────────────────────────────
        with st.expander("🦜 Arten-Katalog", expanded=False):
            catalog = species_load()
            st.caption(f"{len(catalog)} Vogelarten bekannt.")

            with st.form("add_species", clear_on_submit=True, border=False):
                new_name = st.text_input(
                    "Neue Vogelart hinzufügen",
                    placeholder="z. B. Erlenzeisig",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("➕ Hinzufügen", use_container_width=True)
                if submitted:
                    ok, msg = species_add(new_name)
                    if ok:
                        st.toast(f"🦜 {msg}")
                    else:
                        st.warning(msg)

            if catalog:
                with st.popover("Liste anzeigen", use_container_width=True):
                    st.write(", ".join(catalog))

        st.divider()
        st.markdown("##### Hilfe")
        st.markdown(
            """
- 🟢 / 🟡 / 🔴 zeigt die Konfidenz pro Zeile
- **⚓ Anker-Zelle** (erste Bague-Zeile): Änderung rechnet die ganze Bague-Spalte neu
- **⚠ Review-Banner** oben: Cells wo HTR-VT und TrOCR sich uneinig waren
- **Tab / Enter** zur nächsten Zelle (wie in Excel)
- **Unbekannter Vogelname?** Trag ihn in die Espèce-Spalte ein, dann erscheint
  oben ein Hinweis zum Speichern.
            """
        )

    # Immer sichtbar — anmelden geht auch schon vor dem Upload.
    st.divider()
    render_nextcloud_login()


# ──────────────────────────────────────────────────────────────────────
# Main: upload OR editor
# ──────────────────────────────────────────────────────────────────────
if not st.session_state.get("uploaded"):
    st.markdown("# 🦜 DocDig")
    st.markdown(
        "Lade eine Beringungsliste (PDF) hoch. Die App extrahiert die Tabelle, "
        "erkennt Spalten und Werte, und du kannst sie korrigieren bevor du sie als CSV speicherst."
    )
    uploaded = st.file_uploader("Beringungsliste (PDF)", type="pdf", label_visibility="collapsed")
    if not uploaded:
        st.stop()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = INPUT_DIR / uploaded.name
    pdf_path.write_bytes(uploaded.getbuffer())

    with st.spinner("PDF wird in Einzelseiten zerlegt …"):
        base = build_base_pipeline()
        all_pages = base.run(input_data=str(pdf_path))

    st.session_state.uploaded_name = uploaded.name
    st.session_state.pdf_path = str(pdf_path)
    st.session_state.all_pages = all_pages
    st.session_state.predictions = [None] * len(all_pages)
    st.session_state.page_idx = 0
    st.session_state.processed = False
    st.session_state.uploaded = True
    st.rerun()


# Predictions not yet built — show "process" button.
if not st.session_state.get("processed"):
    st.markdown("## 📄 Vorschau")
    pdf_viewer(st.session_state.pdf_path, height=700)
    if st.button(
        "🔍 Alle Seiten verarbeiten",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Pipeline läuft – das kann eine Minute pro Seite dauern …"):
            pipeline = build_page_pipeline(st.session_state.all_pages)
            results = pipeline.run()
            if not isinstance(results, list):
                results = [results]
            st.session_state.predictions = results
            st.session_state.processed = True
        st.rerun()
    st.stop()


# ──────────────────────────────────────────────────────────────────────
# Editor view
# ──────────────────────────────────────────────────────────────────────
page_idx = st.session_state.page_idx
page = st.session_state.predictions[page_idx]
df, visible = page_to_dataframe(page)
column_labels = [label for _, label in visible]
n_pages = len(st.session_state.all_pages)


# ---- per-cell metadata (score, alternatives) lookups ----------------------
def _cell_meta(page, col_i, row):
    cells = page["columns"][col_i].get("cells", [])
    if row >= len(cells):
        return None
    return cells[row]


# Build a (row, label, chosen, alt, score, col_i) review list once — used by
# both the top banner and the row-status column.
review_items = []
high_conf = mid_conf = low_conf = empty = 0
for col_i, label in visible:
    col = page["columns"][col_i]
    for c_idx, cell in enumerate(col.get("cells", [])[1:], start=1):
        score = int(cell.get("score", -1) or -1)
        if not (cell.get("erkannt") or "").strip():
            empty += 1
        elif score >= 95:
            high_conf += 1
        elif score >= 60:
            mid_conf += 1
        elif score >= 0:
            low_conf += 1
        alts = cell.get("alternatives") or []
        if alts and score < 95:
            review_items.append(
                (c_idx, label, cell.get("erkannt", ""), alts[0], score, col_i)
            )

# Document-wide aggregate (for the header progress bar).
# Every extracted column counts — the UI shows them all.
doc_total = doc_high = doc_mid = doc_low = doc_empty = 0
for p in st.session_state.predictions:
    if p is None:
        continue
    for col in p.get("columns", []):
        for cell in col.get("cells", [])[1:]:
            doc_total += 1
            score = int(cell.get("score", -1) or -1)
            if not (cell.get("erkannt") or "").strip():
                doc_empty += 1
            elif score >= 95:
                doc_high += 1
            elif score >= 60:
                doc_mid += 1
            elif score >= 0:
                doc_low += 1


# ===== Compact header: page nav + stats =====
header_cols = st.columns([1, 4, 1])
with header_cols[0]:
    if st.button(
        "◀ Zurück", use_container_width=True, disabled=(page_idx == 0), key="nav_prev_main"
    ):
        st.session_state.page_idx -= 1
        st.rerun()
with header_cols[1]:
    doc_pct = int(round(doc_high / max(doc_total, 1) * 100))
    st.markdown(
        f"<div style='text-align:center;font-size:22px;font-weight:600;margin-top:0.3rem;'>"
        f"Seite {page_idx + 1} von {n_pages}"
        f"</div>"
        f"<div style='text-align:center;font-size:13px;color:#666;margin-top:2px;'>"
        f"Diese Seite: "
        f"<span class='pill pill-ok'>✓ {high_conf}</span>&nbsp;"
        f"<span class='pill pill-warn'>⚠ {mid_conf}</span>&nbsp;"
        f"<span class='pill pill-bad'>! {low_conf}</span>&nbsp;"
        f"<span style='color:#999;'>{empty} leer</span>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;Dokument: <b>{doc_pct}%</b> sicher "
        f"<span style='color:#999;'>({doc_high}/{doc_total})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(doc_pct / 100)
with header_cols[2]:
    if st.button(
        "Weiter ▶",
        use_container_width=True,
        disabled=(page_idx >= n_pages - 1),
        key="nav_next_main",
    ):
        st.session_state.page_idx += 1
        st.rerun()


# ===== Review banner (top) =====
if review_items:
    n = len(review_items)
    with st.expander(
        f"⚠️ {n} Werte zum Überprüfen — beide Erkenner sind uneinig",
        expanded=(n <= 5),
    ):
        st.caption(
            "Pro Zeile: gewählter Wert, Alternative, Score. "
            "Klick auf **↔** übernimmt die Alternative."
        )
        # Two parallel columns of review rows so the user sees more at once.
        n_cols = 2 if n > 6 else 1
        cols = st.columns(n_cols)
        for i, (row, label, chosen, alt, score, col_i) in enumerate(review_items[:30]):
            with cols[i % n_cols]:
                row_cols = st.columns([1, 2, 3, 3, 1], gap="small")
                row_cols[0].markdown(f"**#{row}**")
                row_cols[1].markdown(f"`{label}`")
                row_cols[2].markdown(
                    f"<span class='pill pill-warn'>{score}</span> `{chosen}`",
                    unsafe_allow_html=True,
                )
                row_cols[3].markdown(f"alt: `{alt}`")
                btn_key = f"swap-{page_idx}-{col_i}-{row}"
                if row_cols[4].button("↔", key=btn_key, help=f"Übernehmen: {alt}"):
                    cell = _cell_meta(page, col_i, row)
                    if cell is not None:
                        cell["erkannt"] = alt
                        cell["alternatives"] = [chosen]
                        cell["score"] = max(60, score)
                        mark_manual_edit(cell)
                        st.toast(f"↔ Alternative übernommen: {alt}")
                        st.rerun()


# ===== Unknown-species banner (offer to persist new bird names) =====
pending = [
    item for item in st.session_state.get("pending_species", []) if item[0] == page_idx
]
if pending:
    with st.expander(
        f"🦜 {len(pending)} unbekannte Vogelart(en) auf dieser Seite",
        expanded=True,
    ):
        st.caption(
            "Du hast einen Namen eingetragen, der nicht im Arten-Katalog steht. "
            "Speichere ihn, damit er beim nächsten Scan automatisch erkannt wird."
        )
        # Deduplicate by name (same name might appear in multiple cells)
        seen = set()
        for p_idx, c_idx, r_idx, name in pending:
            if name in seen:
                continue
            seen.add(name)
            cols = st.columns([4, 1, 1])
            cols[0].markdown(f"🦜 `{name}`")
            save_key = f"save-species-{name}"
            ignore_key = f"ignore-species-{name}"
            if cols[1].button("✓ Speichern", key=save_key):
                ok, msg = species_add(name)
                if ok:
                    st.toast(f"🦜 {msg}")
                else:
                    st.warning(msg)
                st.session_state.pending_species = [
                    it for it in st.session_state.pending_species
                    if it[3] != name
                ]
                st.rerun()
            if cols[2].button("✗ Ignorieren", key=ignore_key):
                st.session_state.pending_species = [
                    it for it in st.session_state.pending_species
                    if it[3] != name
                ]
                st.rerun()


# ===== Two-column layout: sticky scan | editable table =====
left, right = st.columns([5, 7], gap="medium")

with left:
    st.markdown("#### Original-Scan")
    page_path = st.session_state.all_pages[page_idx]
    # Height-capped scroll container: scan + table share the screen, and the
    # image's fullscreen lightbox (click to zoom, ✕/Esc to close) keeps
    # working because no ancestor creates a stacking context.
    with st.container(height=760):
        st.image(page_path, use_container_width=True)

with right:
    column_config = {}
    for label in column_labels:
        # Duplicate detections get a " (2)" suffix — configure by base label.
        base = label.split(" (")[0]
        if base == "Bague":
            column_config[label] = st.column_config.TextColumn(
                label if label != base else "Bague No",
                help="⚓ Anker-Zelle: Änderung rechnet die ganze Spalte neu. "
                "Andere Zellen: Werte werden als manuell markiert.",
                max_chars=10,
            )
        elif base == "Espèce":
            column_config[label] = st.column_config.TextColumn(
                label,
                help="Vogelart. Ein \" steht für 'gleich wie oben' (Ditto-Mark).",
            )
        elif base == "Sexe":
            column_config[label] = st.column_config.SelectboxColumn(
                label,
                options=SEXE_OPTIONS,
                help="m=mâle/Männchen, w=Weibchen, f=femelle, (f)=unsicher, X=Markierung, ?=unbekannt",
            )
        elif base == "Age":
            column_config[label] = st.column_config.TextColumn(
                label, help="Alter (z. B. ad., juv., vj.)."
            )
        elif base == "Jour/Mois":
            column_config[label] = st.column_config.TextColumn(
                label, help="Datum als Tag.Monat (z. B. 15.8)."
            )
        elif base == "Heure":
            column_config[label] = st.column_config.TextColumn(
                label, help="Uhrzeit (z. B. 14:30)."
            )
        elif base == "Aile":
            column_config[label] = st.column_config.TextColumn(
                label if label != base else "Aile (mm)", help="Flügellänge in Millimetern."
            )
        elif base == "Poids":
            column_config[label] = st.column_config.TextColumn(
                label if label != base else "Poids (g)", help="Gewicht in Gramm."
            )
        else:
            column_config[label] = st.column_config.TextColumn(
                label, help="Nicht klassifizierte Spalte — Rohinhalt aus dem Scan."
            )

    # Per-row confidence + alternatives flag.
    row_confidence: list[tuple[str, int, bool]] = []  # (emoji, min_score, has_alt)
    for row_offset in range(len(df)):
        row_scores = []
        has_alt = False
        for col_i, _ in visible:
            cell = _cell_meta(page, col_i, row_offset + 1)
            if cell is None:
                continue
            score = int(cell.get("score", -1) or -1)
            if score >= 0:
                row_scores.append(score)
            if cell.get("alternatives"):
                has_alt = True
        if not row_scores:
            row_confidence.append(("⬜", -1, has_alt))
        else:
            ms = min(row_scores)
            if ms >= 95:
                row_confidence.append(("🟢", ms, has_alt))
            elif ms >= 60:
                row_confidence.append(("🟡", ms, has_alt))
            else:
                row_confidence.append(("🔴", ms, has_alt))

    # ── Per-page filter ────────────────────────────────────────────────
    show_only_review = st.session_state.get("show_only_review", False)
    fcols = st.columns([3, 1])
    fcols[0].caption(
        f"💡 Tab/Enter springt zur nächsten Zelle. "
        f"Spalte **📷** zeigt was OCR gesehen hat — perfekt zum Vergleichen."
    )
    show_only_review = fcols[1].toggle(
        "Nur Review-Zellen", value=show_only_review, key="filter_review_toggle"
    )
    st.session_state.show_only_review = show_only_review

    df_with_status = df.copy()
    df_with_status.insert(0, "✓", [c[0] for c in row_confidence])
    if any(c[2] for c in row_confidence):
        df_with_status["⚠"] = ["↔" if c[2] else "" for c in row_confidence]

    if show_only_review:
        mask = [
            (c[0] in ("🔴", "🟡") or c[2])
            for c in row_confidence
        ]
        df_with_status = df_with_status[mask]
        if df_with_status.empty:
            st.success("🎉 Keine Zellen brauchen Review auf dieser Seite.")

    column_config["✓"] = st.column_config.TextColumn(
        "✓",
        help="🟢 hoch  ·  🟡 mittel  ·  🔴 unsicher  ·  ⬜ leer",
        disabled=True,
        width="small",
    )
    if "⚠" in df_with_status.columns:
        column_config["⚠"] = st.column_config.TextColumn(
            "⚠",
            help="↔ in dieser Zeile gibt es eine Alternative — siehe Review-Banner oben",
            disabled=True,
            width="small",
        )

    # Image columns are read-only previews of what OCR actually saw.
    for label in column_labels:
        img_key = f"📷 {label}"
        if img_key in df_with_status.columns:
            column_config[img_key] = st.column_config.ImageColumn(
                "📷",
                help=f"Bild-Crop, das vom OCR für „{label}“ analysiert wurde.",
                width="small",
            )

    editor_key = f"editor-page-{page_idx}-{'review' if show_only_review else 'all'}"
    edited_df = st.data_editor(
        df_with_status,
        column_config=column_config,
        use_container_width=True,
        hide_index=False,
        key=editor_key,
        num_rows="fixed",
        height=min(900, 45 + 36 * len(df_with_status)),
    )

    # Strip the read-only helper columns before syncing.
    if not df_with_status.empty:
        edited_data = edited_df[column_labels].copy()
        df_clean = df.loc[df_with_status.index, column_labels].copy()
        if not edited_data.equals(df_clean):
            # When filtered, edited_data has gaps — merge into the full frame.
            full_after_edit = df[column_labels].copy()
            full_after_edit.loc[df_with_status.index] = edited_data
            changed = sync_edits(page_idx, full_after_edit, visible)
            if changed:
                st.toast(f"✏️ {changed} Zelle(n) aktualisiert.")
            st.rerun()

    # ── Einzel-Ergebnisse der Erkenner (Debug-/Vergleichsansicht) ──────
    _PREDICTION_SOURCES = [
        ("digit", "Tesseract/CRNN"),
        ("yolo", "YOLO"),
        ("trocr", "TrOCR"),
        ("htr_vt", "HTR-VT"),
    ]
    model_rows = []
    for col_i, label in visible:
        cells = page["columns"][col_i].get("cells", [])
        for row_i, cell in enumerate(cells[1:], start=1):
            preds = cell.get("predictions") or {}
            if not preds:
                continue
            model_rows.append(
                {
                    "#": row_i,
                    "Spalte": label,
                    **{
                        col_name: preds.get(key, "")
                        for key, col_name in _PREDICTION_SOURCES
                    },
                    "Final": cell.get("erkannt", ""),
                    "Score": cell.get("score", -1),
                }
            )
    if model_rows:
        with st.expander(
            f"🔬 Einzel-Ergebnisse der Erkennungsmodelle ({len(model_rows)} Zellen)",
            expanded=False,
        ):
            st.caption(
                "Rohe Vorhersage jedes Modells pro Zelle, vor Konsens/Fuzzy. "
                "So siehst du, welches Modell was gelesen hat — Vergleich auf "
                "Fixture-Daten: `python tools/compare_digit_backends.py`. "
                "Backends umschalten: `DOCDIG_DIGIT_BACKENDS=yolo streamlit run src/app.py`."
            )
            model_df = pd.DataFrame(model_rows).sort_values(["Spalte", "#"])
            st.dataframe(model_df, use_container_width=True, hide_index=True)
