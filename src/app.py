"""Streamlit UI for DocDig — bird-banding list digitisation.

Layout:
- Sidebar: file info, CSV export, Nextcloud, catalogs, help.
- Main:    scan panel (own fragment, row-focus crop) | editable table
           (own fragment) — or table-only mode for two-monitor work.

Editing architecture (WICHTIG — do not regress this):
- The editor DataFrame is built ONCE per page and cached in session state.
  `st.data_editor` gets the SAME object every rerun, so the grid keeps its
  scroll position and focused cell.
- Edits flow through the widget's ``edited_rows`` delta in an ``on_change``
  callback (libs/editing.apply_editor_deltas) — there is NO forced full
  rerun per keystroke. The editor lives in a fragment, so its interactions
  never redraw the rest of the page (no jump to top, the scan panel and its
  fullscreen state survive).
- Status columns (✓/⚠) are snapshots from cache-build time; the refresh
  button re-computes them deliberately (updating them per edit would reset
  the grid — Streamlit issue #10181).
- Bague anchor edits are the exception: they rebuild the whole column and
  invalidate the page cache.
"""

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from pathlib import Path

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
from libs.bague_sequence import mark_manual_edit
from libs.column_schema import (
    load_schema,
    reset_schema,
    rows_to_schema,
    save_schema,
    schema_to_rows,
)
from libs.editing import apply_editor_deltas
from libs.table_view import (
    build_csv_bytes,
    compute_row_confidence,
    page_to_dataframe,
)
from libs.value_catalog import (
    add_option as catalog_add_option,
    load_options as catalog_load_options,
    options_with_values as catalog_options_with_values,
    remove_option as catalog_remove_option,
)
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
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
# Static serving (see .streamlit/config.toml) — page scans are copied here so
# they can be opened in a separate browser tab / second monitor.
# Streamlit resolves the static root NEXT TO THE ENTRYPOINT (src/static), and
# serves it under <base>/app/static/... — both verified against a running
# server (the other combinations return the SPA catch-all page or 404).
STATIC_PAGES_DIR = Path("src/static/pages")


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
# Editor page-view cache
# ──────────────────────────────────────────────────────────────────────
# One entry per page: the editor DataFrame (incl. thumbnails — expensive!),
# status snapshot and review list. Built once, reused across reruns so the
# data_editor keeps scroll + focus. Invalidated explicitly (refresh button,
# anchor rebuild, reprocessing).


def get_page_view(page_idx: int) -> dict:
    cache = st.session_state.setdefault("editor_cache", {})
    entry = cache.get(page_idx)
    if entry is not None:
        return entry

    version = st.session_state.setdefault("editor_cache_version", {}).get(page_idx, 0)
    page = st.session_state.predictions[page_idx]
    df, visible = page_to_dataframe(page, include_thumbnails=True)
    row_conf = compute_row_confidence(page, visible, len(df))

    df_status = df.copy()
    df_status.insert(0, "✓", [c[0] for c in row_conf])
    if any(c[2] for c in row_conf):
        df_status["⚠"] = ["↔" if c[2] else "" for c in row_conf]

    # Review list: cells where recognizers disagreed (alternatives present).
    review_items = []
    high = mid = low = empty = 0
    for col_i, label in visible:
        col = page["columns"][col_i]
        for c_idx, cell in enumerate(col.get("cells", [])[1:], start=1):
            score = int(cell.get("score", -1) or -1)
            if not (cell.get("erkannt") or "").strip():
                empty += 1
            elif score >= 95:
                high += 1
            elif score >= 60:
                mid += 1
            elif score >= 0:
                low += 1
            alts = cell.get("alternatives") or []
            if alts and score < 95:
                review_items.append(
                    (c_idx, label, cell.get("erkannt", ""), alts[0], score, col_i)
                )

    entry = {
        "version": version,
        "df": df,
        "df_status": df_status,
        "visible": visible,
        "labels": [label for _, label in visible],
        "row_conf": row_conf,
        "review_items": review_items,
        "stats": {"high": high, "mid": mid, "low": low, "empty": empty},
        # Set at render time to the CURRENTLY displayed row index (filtering!)
        # — the on_change callback maps edited_rows positions through this.
        "display_index": list(df.index),
    }
    cache[page_idx] = entry
    return entry


def refresh_page_view(page_idx: int) -> None:
    """Drop the cached view — the next build gets a new editor key (version),
    which deliberately resets the grid (used after anchor rebuilds and for
    the explicit refresh button)."""
    st.session_state.setdefault("editor_cache", {}).pop(page_idx, None)
    versions = st.session_state.setdefault("editor_cache_version", {})
    versions[page_idx] = versions.get(page_idx, 0) + 1


def _on_editor_change(page_idx: int, editor_key: str) -> None:
    """data_editor on_change: apply the edited_rows delta to the predictions.

    Runs before the fragment rerun. NO st.rerun() here — the grid keeps its
    state; predictions are the single source of truth for exports.
    """
    state = st.session_state.get(editor_key)
    entry = st.session_state.get("editor_cache", {}).get(page_idx)
    if not state or entry is None:
        return
    deltas = getattr(state, "edited_rows", None) or state.get("edited_rows") or {}
    if not deltas:
        return

    page = st.session_state.predictions[page_idx]
    results = apply_editor_deltas(
        page,
        deltas,
        displayed_index=entry["display_index"],
        visible=entry["visible"],
        species_catalog=st.session_state.get("species_catalog_cache"),
    )

    pending = st.session_state.setdefault("pending_species", [])
    for r in results:
        if r.unknown_species:
            key = (page_idx, r.unknown_species)
            if key not in pending:
                pending.append(key)
        if r.anchor_rebuilt and r.rebuild_stats:
            # Column values changed wholesale — rebuild the view (new grid).
            st.session_state["_anchor_rebuild_toast"] = r.rebuild_stats
            refresh_page_view(page_idx)


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
            csv_bytes = build_csv_bytes(st.session_state.predictions)
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

        # ── Auswahllisten (Sexe, Age) verwalten ────────────────────────
        def _render_option_catalog(name: str, icon: str, title: str, hint: str):
            with st.expander(f"{icon} {title}", expanded=False):
                options = catalog_load_options(name)
                st.caption(hint)
                with st.form(f"add_{name}_option", clear_on_submit=True, border=False):
                    new_opt = st.text_input(
                        "Neuen Eintrag hinzufügen",
                        placeholder="Neuer Eintrag …",
                        label_visibility="collapsed",
                    )
                    if st.form_submit_button("➕ Hinzufügen", use_container_width=True):
                        ok, msg = catalog_add_option(name, new_opt)
                        st.toast(f"{icon} {msg}") if ok else st.warning(msg)
                        if ok:
                            refresh_page_view(st.session_state.get("page_idx", 0))
                removable = [o for o in options if o]
                if removable:
                    rm_cols = st.columns([3, 1])
                    to_remove = rm_cols[0].selectbox(
                        "Eintrag entfernen",
                        removable,
                        label_visibility="collapsed",
                        key=f"rm-select-{name}",
                    )
                    if rm_cols[1].button("🗑", key=f"rm-{name}-option"):
                        ok, msg = catalog_remove_option(name, to_remove)
                        st.toast(f"{icon} {msg}") if ok else st.warning(msg)
                        if ok:
                            refresh_page_view(st.session_state.get("page_idx", 0))
                            st.rerun()

        _render_option_catalog(
            "sexe", "⚥", "Sexe-Auswahl",
            "Einträge der Sexe-Auswahlliste im Editor. Eigene Kürzel wie "
            "„(m)?“ einfach ergänzen.",
        )
        _render_option_catalog(
            "age", "🎂", "Age-Auswahl",
            "Alterscodes für Auswahlliste UND Erkennung. Auf den "
            "1972er-Formularen nur Fd/Fnd — weitere Codes (ad., juv., …) "
            "hier ergänzen.",
        )

        st.divider()
        st.markdown("##### Hilfe")
        st.markdown(
            """
- 🟢 / 🟡 / 🔴 zeigt die Konfidenz pro Zeile (Stand beim Seitenaufbau —
  der 🔄-Button über der Tabelle berechnet sie neu)
- **⚓ Anker-Zelle** (Bague): Änderung rechnet die ganze Spalte neu —
  Buchstaben-Präfixe wie `A90401` bleiben erhalten
- **Ditto („gleich wie oben“)**: einfach `"` in die Zelle tippen
- **Tab / Enter** zur nächsten Zelle (wie in Excel)
- **Zweiter Bildschirm**: „Scan in neuem Tab öffnen“ über dem Scan-Panel,
  dazu Ansicht „Nur Tabelle“
- **Unbekannter Vogelname?** Eintragen → oben erscheint ein Hinweis zum
  Speichern in den Katalog.
            """
        )

    # Immer sichtbar — anmelden geht auch schon vor dem Upload.
    st.divider()
    render_nextcloud_login()


def render_schema_editor():
    """Spalten vordefinieren — BEFORE processing.

    Edits are saved to config/column_schema.json; the pipeline reads the
    file when '🔍 Alle Seiten verarbeiten' is clicked, so changes made here
    take effect for the next run.
    """
    with st.expander("🧩 Spalten-Layout des Formulars (vordefiniert)", expanded=False):
        st.caption(
            "So erwartet die Erkennung die Spalten — in dieser Reihenfolge. "
            "**Pos** ändern zum Umsortieren, **Aktiv** abwählen wenn das "
            "Formular eine Spalte nicht hat, **Stichwörter** = Begriffe der "
            "gedruckten Kopfzeile (Kommas trennen). Breiten in % der "
            "Tabellenbreite. Gilt ab der nächsten Verarbeitung."
        )
        schema = load_schema()
        rows = schema_to_rows(schema)

        edited = st.data_editor(
            pd.DataFrame(rows).drop(columns=["_role"]),
            key="schema-editor",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Pos": st.column_config.NumberColumn(
                    "Pos", min_value=1, max_value=len(rows), step=1, width="small"
                ),
                "Spalte": st.column_config.TextColumn("Spalte", disabled=True),
                "Aktiv": st.column_config.CheckboxColumn("Aktiv", width="small"),
                "Header-Stichwörter": st.column_config.TextColumn(
                    "Header-Stichwörter (Komma-getrennt)"
                ),
                "Breite min %": st.column_config.NumberColumn(
                    "Breite min %", min_value=0.0, max_value=100.0, width="small"
                ),
                "Breite max %": st.column_config.NumberColumn(
                    "Breite max %", min_value=0.0, max_value=100.0, width="small"
                ),
            },
        )

        cols = st.columns([1, 1, 3])
        if cols[0].button("💾 Speichern", key="schema-save", type="primary"):
            merged = edited.copy()
            merged["_role"] = [r["_role"] for r in rows]
            new_schema = rows_to_schema(merged.to_dict("records"), base_schema=schema)
            save_schema(new_schema)
            st.toast("🧩 Spalten-Layout gespeichert — gilt ab der nächsten Verarbeitung.")
            st.rerun()
        if cols[1].button("↩︎ Standard", key="schema-reset"):
            reset_schema()
            st.toast("🧩 Auf Standard-Layout zurückgesetzt.")
            st.rerun()


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
    render_schema_editor()
    if not uploaded:
        st.stop()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = INPUT_DIR / uploaded.name
    pdf_path.write_bytes(uploaded.getbuffer())

    with st.spinner("PDF wird in Einzelseiten zerlegt …"):
        base = build_base_pipeline()
        all_pages = base.run(input_data=str(pdf_path))

    # Copy the page scans into the statically served dir so each page can be
    # opened in its own browser tab (second-monitor workflow).
    import shutil

    static_dir = STATIC_PAGES_DIR / Path(uploaded.name).stem
    static_dir.mkdir(parents=True, exist_ok=True)
    static_urls = []
    for i, page_file in enumerate(all_pages):
        target = static_dir / f"page_{i + 1}{Path(page_file).suffix or '.jpg'}"
        try:
            shutil.copyfile(page_file, target)
            # URL path: src/static/... on disk -> app/static/... over HTTP.
            # Relative href, so it also works behind a proxy base path.
            static_urls.append(f"app/{target.relative_to('src').as_posix()}")
        except Exception:
            static_urls.append(None)
    st.session_state.page_static_urls = static_urls

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
    render_schema_editor()
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
n_pages = len(st.session_state.all_pages)

# Catalog snapshot for the edit callback (cheap file read).
st.session_state["species_catalog_cache"] = species_load()

view = get_page_view(page_idx)
review_items = view["review_items"]
high_conf = view["stats"]["high"]
mid_conf = view["stats"]["mid"]
low_conf = view["stats"]["low"]
empty = view["stats"]["empty"]


# ---- per-cell metadata (score, alternatives) lookups ----------------------
def _cell_meta(page, col_i, row):
    cells = page["columns"][col_i].get("cells", [])
    if row >= len(cells):
        return None
    return cells[row]

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
        refresh_page_view(st.session_state.page_idx)
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
        refresh_page_view(st.session_state.page_idx)
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
                        refresh_page_view(page_idx)
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
        for _p_idx, name in pending:
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
                    if it[1] != name
                ]
                st.rerun()
            if cols[2].button("✗ Ignorieren", key=ignore_key):
                st.session_state.pending_species = [
                    it for it in st.session_state.pending_species
                    if it[1] != name
                ]
                st.rerun()


# ===== Fragments: scan panel | editable table =====
# Each panel is its own st.fragment — interacting with one never redraws the
# other (or the rest of the page). This is what keeps editing fast and stops
# the old "jumps to top / loses the focused cell / closes fullscreen" pain.


@st.cache_resource(show_spinner=False)
def _load_page_image(path: str) -> Image.Image:
    return Image.open(path).copy()


@st.fragment
def render_scan_panel(page_idx: int, num_rows: int) -> None:
    page_path = st.session_state.all_pages[page_idx]

    head = st.columns([3, 2], vertical_alignment="bottom")
    head[0].markdown("#### Original-Scan")
    static_urls = st.session_state.get("page_static_urls") or []
    url = static_urls[page_idx] if page_idx < len(static_urls) else None
    if url:
        head[1].markdown(
            f'<div style="text-align:right;"><a href="{url}" target="_blank">'
            f"🖼 In neuem Tab öffnen</a></div>",
            unsafe_allow_html=True,
        )

    full_view = st.toggle(
        "Ganze Seite",
        value=False,
        key=f"scan-full-{page_idx}",
        help="Aus: Ausschnitt um die gewählte Tabellenzeile (Regler unten).",
    )

    if full_view or num_rows < 2:
        with st.container(height=760):
            st.image(page_path, use_container_width=True)
        return

    # Row focus: the printed forms have uniformly spaced rows, so a
    # proportional band crop follows the table rows well enough to keep the
    # scan aligned with where the user is editing.
    row = st.slider(
        "Zeile im Scan",
        min_value=1,
        max_value=num_rows,
        key=f"scan-row-{page_idx}",
        help="Zeigt den Scan-Ausschnitt um diese Tabellenzeile.",
    )
    img = _load_page_image(page_path)
    w, h = img.size
    table_top, table_bottom = 0.08, 0.985  # header block above, margin below
    band = (table_bottom - table_top) / num_rows
    half_window = band * 5  # ±5 Zeilen sichtbar
    center = table_top + (row - 0.5) * band
    y0 = max(0.0, center - half_window)
    y1 = min(1.0, center + half_window)
    crop = img.crop((0, int(y0 * h), w, int(y1 * h)))
    st.image(crop, use_container_width=True)
    st.caption(f"Ausschnitt um Zeile {row} (±5 Zeilen).")


def _build_column_config(view: dict) -> dict:
    column_config = {}
    for label in view["labels"]:
        # Duplicate detections get a " (2)" suffix — configure by base label.
        base = label.split(" (")[0]
        if base == "Bague":
            column_config[label] = st.column_config.TextColumn(
                label if label != base else "Bague No",
                help="⚓ Anker-Zelle: Änderung rechnet die ganze Spalte neu "
                "(Buchstaben-Präfix wie A90401 bleibt erhalten). "
                "Andere Zellen: Werte werden als manuell markiert.",
                max_chars=12,
            )
        elif base == "Espèce":
            column_config[label] = st.column_config.TextColumn(
                label,
                help="Vogelart. Ein \" steht für 'gleich wie oben' (Ditto-Mark).",
            )
        elif base == "Sexe":
            present = view["df"][label].tolist() if label in view["df"] else []
            column_config[label] = st.column_config.SelectboxColumn(
                label,
                options=catalog_options_with_values("sexe", present),
                help="Auswahlliste ist erweiterbar: Sidebar → „Sexe-Auswahl“.",
            )
        elif base == "Age":
            present = view["df"][label].tolist() if label in view["df"] else []
            column_config[label] = st.column_config.SelectboxColumn(
                label,
                options=catalog_options_with_values("age", present),
                help="Alterscode. Auswahlliste erweiterbar: Sidebar → „Age-Auswahl“.",
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

    column_config["✓"] = st.column_config.TextColumn(
        "✓",
        help="🟢 hoch · 🟡 mittel · 🔴 unsicher · ⬜ leer — Schnappschuss, "
        "🔄 berechnet neu",
        disabled=True,
        width="small",
    )
    column_config["⚠"] = st.column_config.TextColumn(
        "⚠",
        help="↔ es gibt eine Alternative — siehe Review-Banner oben",
        disabled=True,
        width="small",
    )
    for label in view["labels"]:
        img_key = f"📷 {label}"
        column_config[img_key] = st.column_config.ImageColumn(
            "📷",
            help=f"Bild-Crop, das vom OCR für „{label}“ analysiert wurde.",
            width="small",
        )
    return column_config


@st.fragment
def render_editor_panel(page_idx: int) -> None:
    view = get_page_view(page_idx)
    page = st.session_state.predictions[page_idx]

    rebuild_stats = st.session_state.pop("_anchor_rebuild_toast", None)
    if rebuild_stats:
        st.toast(
            f"🔄 Bague-Spalte aus Anker {rebuild_stats['anchor']} neu berechnet "
            f"({rebuild_stats['filled']} Zellen)."
        )

    fcols = st.columns([5, 2, 1], vertical_alignment="center")
    fcols[0].caption(
        "💡 Tab/Enter wie in Excel · `\"` = Ditto · ✓/⚠ sind ein "
        "Schnappschuss — 🔄 berechnet sie neu."
    )
    show_only_review = fcols[1].toggle(
        "Nur Review-Zellen", key=f"filter-review-{page_idx}"
    )
    if fcols[2].button(
        "🔄",
        key=f"refresh-view-{page_idx}",
        help="Statusspalten, Review-Banner und Fortschritt neu berechnen.",
    ):
        refresh_page_view(page_idx)
        st.rerun()  # voller Rerun: auch Banner/Pills oben aktualisieren

    df_display = view["df_status"]
    if show_only_review:
        mask = [(c[0] in ("🔴", "🟡") or c[2]) for c in view["row_conf"]]
        df_display = df_display[pd.Series(mask, index=df_display.index)]
        if df_display.empty:
            st.success("🎉 Keine Zellen brauchen Review auf dieser Seite.")

    # The on_change callback maps edited_rows positions through this index —
    # it must always reflect the CURRENTLY displayed subset.
    view["display_index"] = list(df_display.index)

    editor_key = (
        f"editor-p{page_idx}-v{view['version']}-{'r' if show_only_review else 'a'}"
    )
    st.data_editor(
        df_display,
        column_config=_build_column_config(view),
        use_container_width=True,
        hide_index=False,
        key=editor_key,
        num_rows="fixed",
        height=min(900, 45 + 36 * max(len(df_display), 1)),
        on_change=_on_editor_change,
        args=(page_idx, editor_key),
    )

    # ── Einzel-Ergebnisse der Erkenner (Debug-/Vergleichsansicht) ──────
    _PREDICTION_SOURCES = [
        ("digit", "Tesseract/CRNN"),
        ("yolo", "YOLO"),
        ("trocr", "TrOCR"),
        ("htr_vt", "HTR-VT"),
        ("sexe_cnn", "Sexe-CNN"),
    ]
    model_rows = []
    for col_i, label in view["visible"]:
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
                "Vergleich auf Fixture-Daten: `python tools/compare_digit_backends.py`. "
                "Backends umschalten: `DOCDIG_DIGIT_BACKENDS=yolo streamlit run src/app.py`."
            )
            model_df = pd.DataFrame(model_rows).sort_values(["Spalte", "#"])
            st.dataframe(model_df, use_container_width=True, hide_index=True)


# ===== Layout: geteilt (Scan | Tabelle) oder nur Tabelle (2. Monitor) =====
layout_mode = st.radio(
    "Ansicht",
    options=["Geteilt", "Nur Tabelle"],
    horizontal=True,
    key="layout-mode",
    label_visibility="collapsed",
    help="„Nur Tabelle“ + Scan im eigenen Browser-Tab = zwei Bildschirme.",
)

num_rows = len(get_page_view(page_idx)["df"])

if layout_mode == "Nur Tabelle":
    static_urls = st.session_state.get("page_static_urls") or []
    url = static_urls[page_idx] if page_idx < len(static_urls) else None
    if url:
        st.markdown(
            f'<a href="{url}" target="_blank">🖼 Scan (Seite {page_idx + 1}) '
            f"in neuem Tab öffnen</a> — fürs Arbeiten mit zwei Bildschirmen.",
            unsafe_allow_html=True,
        )
    render_editor_panel(page_idx)
else:
    left, right = st.columns([5, 7], gap="medium")
    with left:
        render_scan_panel(page_idx, num_rows)
    with right:
        render_editor_panel(page_idx)
