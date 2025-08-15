import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
from PIL import Image
from functools import partial

import random
import base64

from pipeline.cv_pipeline import CVPipeline
from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.column_extractor import ColumnExtractor
from modules.row_extractor import RowExtractor
from modules.detect_columns import DetectColumns
from modules.reorder_columns import ReorderColumns
from modules.cell_denoiser import CellDenoiser
from modules.cell_formatter import CellFormatter
from modules.quotation_mark_detector import QuotationMarkDetector
from modules.trocr import TrOCR
from modules.fuzzy_matching import FuzzyMatchingBirdNames
from modules.fuzzy_matching import FuzzyMatchingAge
from modules.predictor_dummy import PredictorDummy

def get_base64_image(path):
    with open(path, "rb") as img_file:
        data = img_file.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("src/background.png")

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stTextInput > div > div > input {
        padding-top: 6px;
        padding-bottom: 6px;
    }
    [data-testid="stColumn"]:has(div.custom-marker):not(:last-child) { 
        padding: 0px 10px;
        border-right: 1px solid #e0e0e0;
    }

    .stPopover {
        margin-top: 10px;
        padding: 0 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
    .custom-header {{
        background: linear-gradient(to bottom, rgba(255,255,255,0.0), rgba(255,255,255,1)),
                    url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: top center;
        height: 600px;
        width: 100%;
    }}
    </style>

    <div class="custom-header"></div>
    """,
    unsafe_allow_html=True
)

def _reset_session_state():
    if "all_pages" in st.session_state:
        del st.session_state["all_pages"]

    if "page_idx" in st.session_state:
        del st.session_state["page_idx"]

    if "predictions" in st.session_state:
        del st.session_state["predictions"]
    
    if "processing" in st.session_state:
        del st.session_state["processing"]

    if "uploaded_name" in st.session_state:
        del st.session_state["uploaded_name"]

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False

if not st.session_state.uploaded:
    uploaded = st.file_uploader(label="File upload", type="pdf", label_visibility="hidden")

    if not uploaded:
        st.stop()
    
    input_dir = os.path.join("data", "input")
    os.makedirs(input_dir, exist_ok=True)
    pdf_path = os.path.join(input_dir, uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    _reset_session_state()
    if "all_pages" not in st.session_state:
        base_pipeline = CVPipeline(input_data={})
        base_pipeline.add_stage(PdfConverter(debug=False))
        
        st.session_state.all_pages = base_pipeline.run(input_data=pdf_path)
        st.session_state.predictions = [None] * len(st.session_state.all_pages)
        st.session_state.page_idx = 0
        st.session_state.processing = [False] * len(st.session_state.all_pages)
        st.session_state.uploaded_name = uploaded.name
    
    st.session_state.uploaded = True
    st.rerun()

idx = st.session_state.page_idx
all_pages = st.session_state.all_pages
page_path = all_pages[idx]

if st.session_state.predictions[idx] is None and not st.session_state.processing[idx]:

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(
        page_path,
        caption=f"Original Seite {idx+1}/{len(all_pages)}",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Seite verarbeiten", use_container_width=True):
        st.session_state.processing[idx] = True
        st.rerun()

elif st.session_state.processing[idx]:

    st.markdown(
        "<div style='text-align:center; font-size: 24px;'>⏳ Seite wird verarbeitet...</div>",
        unsafe_allow_html=True,
    )

    page_input = {"pdf-converter": [page_path]}
    page_pipeline = CVPipeline(input_data=page_input)
    page_pipeline.add_stage(TableRotator(debug=False))
    page_pipeline.add_stage(TatrExtractor(debug=False))
    page_pipeline.add_stage(ColumnExtractor(debug=False))
    page_pipeline.add_stage(RowExtractor(debug=False))
    page_pipeline.add_stage(DetectColumns())
    page_pipeline.add_stage(ReorderColumns())
    page_pipeline.add_stage(CellDenoiser(debug=False))

    page_pipeline.add_stage(CellFormatter())
    page_pipeline.add_stage(QuotationMarkDetector())
    page_pipeline.add_stage(TrOCR())
    page_pipeline.add_stage(FuzzyMatchingBirdNames())
    page_pipeline.add_stage(FuzzyMatchingAge())

    result = page_pipeline.run()
    #print(f"ResultsLen: {len(result)}\nResultsLen result['columns'] {len(result['columns'])}")

    st.session_state.predictions[idx] = result
    st.session_state.processing[idx] = False
    st.rerun()

else:
    pred = st.session_state.predictions[idx]

    # Unpack single-page result if needed
    if isinstance(pred, list) and len(pred) == 1:
        pred = pred[0]

    columns = pred["columns"]
    cells_only = [col["cells"] for col in columns]
    rows = list(zip(*cells_only))

    st.markdown("## 🧾 Erkannte Zellstruktur")

    # ---- HEADER (row 0) ----
    header_row = rows[0]
    with st.container(border=True):
        header_cols = st.columns(len(header_row), vertical_alignment="center", gap=None)
        for col_idx, cell in enumerate(header_row):
            with header_cols[col_idx]:
                st.markdown('<div class="custom-marker"></div>', unsafe_allow_html=True)

                img_array = cell["image"]
                if img_array is None:
                    st.markdown('<span style="color:red">Cell Not Detected</span>', unsafe_allow_html=True)
                else:
                    img_uint8 = (
                        (img_array * 255).astype("uint8")
                        if img_array.max() <= 1
                        else img_array.astype("uint8")
                    )
                    st.image(Image.fromarray(img_uint8), use_container_width=True)

    # ---- OTHER ROWS (starting from row 1) ----
    for row_idx, row in enumerate(rows[1:], start=1):
        with st.container(border=True):

            # Display Images
            image_cols = st.columns(len(row), vertical_alignment="center", gap=None)
            for col_idx, cell in enumerate(row):
                with image_cols[col_idx]:
                    st.markdown('<div class="custom-marker"></div>', unsafe_allow_html=True)

                    img_array = cell["image"]
                    if img_array is None:
                        st.markdown('<span style="color:red">Cell Not Detected</span>', unsafe_allow_html=True)
                    else:
                        img_uint8 = (
                            (img_array * 255).astype("uint8")
                            if img_array.max() <= 1
                            else img_array.astype("uint8")
                        )
                        st.image(Image.fromarray(img_uint8), use_container_width=True)

            #erkannt_cols = st.columns(len(row))

            # Display recognized text
            ##for col_idx, cell in enumerate(row):
            ##    with erkannt_cols[col_idx]:
            ##        erkannt_text = cell.get("erkannt", "")
            ##        score = cell.get("score", -1)
    ##
            ##        if score >= 90:
            ##            color = "#d4edda"  # green
            ##        elif score >= 50:
            ##            color = "#fff3cd"  # yellow
            ##        else:
            ##            color = "#f8d7da"  # red
    ##
            ##        st.markdown(
            ##            f"""
            ##            <div style='background-color:{color}; color: black; padding:6px; border-radius:6px; text-align:center;'>
            ##                <b>Erkannt:</b> {erkannt_text}
            ##            </div>
            ##            """,
            ##            unsafe_allow_html=True,
            ##        )

            # Input field for correction
            input_cols = st.columns(len(row), vertical_alignment="center", gap=None)
            for col_idx, cell in enumerate(row):
                with input_cols[col_idx]:
                    erkannt_text = cell.get("erkannt", None)

                    st.markdown(
                        f"""
                        <div style='text-align:center; font-size:12px; font-weight: bold; padding: 0px 8px; border-right: 1px solid #e0e0e0;'>
                            {erkannt_text if erkannt_text else ' - '}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.popover("", icon=":material/edit:", use_container_width=True):
                        cell_key = f"cell-{idx}-{row_idx}-{col_idx}"
                        
                        def on_change(_cell_key, _idx, _col_idx, _row_idx):
                            st.session_state.predictions[_idx][0]["columns"][_col_idx]["cells"][_row_idx]["erkannt"] = st.session_state[_cell_key]
                            #st.rerun()

                        st.text_input(
                            "Verbesserung:",
                            key=cell_key,
                            value=erkannt_text,
                            label_visibility="collapsed",
                            placeholder="Enter",
                            on_change=partial(on_change, cell_key, idx, col_idx, row_idx),
                        )
                    #cell_key = f"verbesserung-{idx}-{row_idx}-{col_idx}"
#
                    #def on_change(_cell_key, _idx, _col_idx, _row_idx):
                    #    st.session_state.predictions[_idx][0]["columns"][_col_idx]["cells"][_row_idx]["erkannt"] = st.session_state[_cell_key]
                    #    st.rerun()
#
                    #st.text_input(
                    #    "Verbesserung:",
                    #    key=cell_key,
                    #    value=erkannt_text,
                    #    label_visibility="collapsed",
                    #    placeholder="Enter",
                    #    on_change=partial(on_change, cell_key, idx, col_idx, row_idx),
                    #)

    if st.session_state.page_idx + 1 < len(all_pages):
        if st.button("➡️ Nächste Seite"):
            st.session_state.page_idx += 1
            st.rerun()
    else:
        if st.button("Finalisieren und Download"):
            # FIXME: Move to dedicated module...
            import csv

            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{st.session_state.uploaded_name}_predictions.csv")

            with open(output_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                for idx in range(len(st.session_state.predictions)):
                    page_data = st.session_state.predictions[idx][0]["columns"]
                    num_rows = len(page_data[0]["cells"])

                    if idx == 0:
                        row = []
                        for col in page_data:
                            if col["is_batch_column"]:
                                row.append("Batch")
                            elif col["is_species_column"]:
                                row.append("Species")
                            elif col["is_sexe_column"]:
                                row.append("Sexe")
                            elif col["is_age_column"]:
                                row.append("Age")
                            elif col["is_jour-mois_column"]:
                                row.append("Jour-Mois")
                            elif col["is_heure_column"]:
                                row.append("Heure")
                            elif col["is_alle_column"]:
                                row.append("Alle")
                            elif col["is_poids_column"]:
                                row.append("Poids")
                            else:
                                row.append("Unknown")

                        writer.writerow(row)
                        writer.writerow([])

                    
                    writer.writerow([f"Page {idx + 1}"])

                    for row_idx in range(1, num_rows):
                        row = [
                            page_data[col_idx]["cells"][row_idx]["erkannt"]
                            for col_idx in range(len(page_data))
                        ]
                        writer.writerow(row)

                    writer.writerow([])

            st.markdown(
                f'<a href="data:file/csv;base64,{base64.b64encode(open(output_path, "rb").read()).decode()}" download="{st.session_state.uploaded_name}_predictions.csv">Download Predictions</a>',
                unsafe_allow_html=True,
            )
        
        if st.button("Zurücksetzen", type="primary"):
            _reset_session_state()
            st.session_state.uploaded = False
            
            st.rerun()
