## Prerequisites

- Virtual Environment with Python < 3.13 is needed, to make tensorflow work
- Always execute the main script from the root folder, aka `python3 src/main.py`!
- Start WebApp with:

```bash
streamlit run src/app.py
```

## Modelle — was liegt wo

Canonical location for every model is `./config/` — the pipeline resolves
all weights from there. Policy: small/medium trained models are tracked in
git (like the pre-existing `classifier_model.keras`); the two 408 MB HTR-VT
checkpoints are NOT in git (`*.pth` is gitignored) and are distributed via
the Vogelwarte Nextcloud (`/docdig/models/`).

| File | Size | In git? | In Docker image? | Source |
|---|---|---|---|---|
| `config/digit_model.keras` | 3 MB | ja | ja | `tools/train_digit_model.py` (DIDA) |
| `config/digit_yolo.pt` | 18 MB | ja | ja | YOLOv8, `ml_spielereien/runs/detect/train5` |
| `config/sexe_model.keras` | 65 MB | ja | ja | `tools/train_sexe_model.py` (M-W-dataset) |
| `config/sexe_classes.json` | <1 KB | ja | ja | auto-written by the training script |
| `config/classifier_model.keras` | 7 MB | ja | ja | legacy species classifier |
| `config/denoise_model.keras` | 15 MB | ja | ja | only needed if CellDenoiser is re-enabled |
| `config/htr_vt/best_WER.pth` | 408 MB | **nein** | **nein** | Nextcloud `/docdig/models/htr_vt/` — mount as volume |
| `config/htr_vt/alphabet.json` | <1 KB | ja | ja | cached alphabet for HTR-VT |
| `config/pubtables1m_*.pth` (TATR) | 2×110 MB | nein | ja (downloaded at build) | HuggingFace, see `docker/Dockerfile` |

Without `best_WER.pth` the app still runs — the Espèce column falls back to
TrOCR. Everything else degrades the same way: missing model = no-op stage.

## Required configuration files

The pipeline expects these files under `./config/` (see also `link_to_tatr_models.txt`):

- `detection_config.json`, `structure_config.json` — TATR configs
- `pubtables1m_detection_detr_r18.pth`, `pubtables1m_structure_detr_r18.pth` — TATR weights
- `class_indices.json` — bird species labels (used by fuzzy matching)
- `age_classes.json` — age labels

Optional, but recommended:

- `htr_vt/best_WER.pth` (and/or `best_CER.pth`) — HTR-VT checkpoint for the species column
- `htr_vt/train.ln` + `htr_vt/lines/` — used **once** to derive the HTR-VT alphabet; the
  result is cached to `config/htr_vt/alphabet.json` and afterwards the raw files
  are no longer needed. (Equivalent files exist under `FelixUpload/inferencedata/digC/`
  and are auto-discovered if `config/htr_vt/` is empty.)
- `digit_model.keras` — Keras CRNN+CTC for the bague column (input shape
  `(40, 160, 1)`, CTC head over `"0123456789"`). Train with
  `python tools/train_digit_model.py`.
- `digit_yolo.pt` — YOLOv8 per-digit detector (10 classes, "0"–"9"; trained in
  `ml_spielereien/runs/detect/train5`). Third voice of the numeric ensemble
  next to Tesseract/CRNN and TrOCR. Requires the `ultralytics` package.
- `sexe_model.keras` — Keras CNN for the sexe column (input shape
  `(150, 60, 1)`, N-class softmax over `m` / `none` / `w`). Train with
  `python tools/train_sexe_model.py` — requires `M-W-Classification/dataset/`
  with one subdirectory per class.
- `sexe_classes.json` — list of class labels matching the softmax order
  (auto-written by the training script).
- `denoise_model.keras` — only required if you re-enable `CellDenoiser`
  (dropped from the default pipeline because it hurt downstream OCR).

If a model file is missing, the corresponding recognizer becomes a no-op.
TrOCR (`microsoft/trocr-base-stage1`) runs on the Espèce column and on every
numeric column (Aile, Poids, Heure, Jour/Mois).

## Pipeline overview

```
PdfConverter
    -> TableRotator
    -> TatrExtractor
    -> MergedColumnExtractor   (global column template)
    -> MergedRowExtractor      (global row template -> cell crops)
    -> DetectColumns           (tags columns: batch / species / sexe / age / …)
    -> CellFormatter           (schema + BLANK GATE: inkless cells get
                                is_blank + skip_ocr — no recognizer ever
                                hallucinates into an empty cell)
    -> QuotationMarkDetector   (detects "ditto" cells, sets skip_ocr;
                                blank cells stay empty, not ditto)
    -> HtrVtRecognizer         (species + age columns, uses image_raw)
    -> DigitRecognizer         (bague sequence + every numeric column;
                                Tesseract/CRNN + YOLO per-digit detector)
    -> SexeClassifier          (sexe column, uses image_raw)
    -> TrOCR                   (Espèce + all numeric columns)
    -> FuzzyMatchingBirdNames  (snaps species to class_indices.json)
    -> NumericConsensus        (majority voting: digit vs. yolo vs. trocr)
    -> FuzzyMatchingAge        (snaps age to age_classes.json)
```

All recognizers dispatch on the semantic flags set by `DetectColumns`
(`is_species_column`, `is_batch_column`, …) — not on column indices, so adding
or reordering columns in the source documents does not require code changes.
The flags and their labels live in one place: `src/libs/columns.py`.

The Streamlit UI shows **every extracted column**: semantically tagged ones
(Bague, Espèce, Sexe, Age, Jour/Mois, Heure, Aile, Poids) with their form
label, anything the detector could not classify as "Spalte N".

## Numeric ensemble

Numbers appear in five columns (Bague, Aile, Poids, Heure, Jour/Mois) and up
to three independent recognizers read each cell:

1. **Tesseract / Keras CRNN** (`predictions["digit"]`) — DigitRecognizer picks
   the more plausible of the two per cell.
2. **YOLOv8 per-digit detector** (`predictions["yolo"]`) — detects individual
   digits, composes them left-to-right (`libs/yolo_digits.py`).
3. **TrOCR** (`predictions["trocr"]`) — generic handwriting OCR.

Every numeric crop is **border-stripped first** (`libs/cell_cropping.py`) —
vertical slivers of the printed table grid read as "1" in every backend
otherwise. Only near-full-height strokes close to the cell edge are removed,
so a handwritten "1" survives.

Selecting backends (e.g. YOLO only):

```bash
DOCDIG_DIGIT_BACKENDS=yolo streamlit run src/app.py       # nur YOLO
DOCDIG_DIGIT_BACKENDS=tesseract,yolo streamlit run ...    # ohne CRNN
```

Comparing them on labelled fixture cells (prints a per-cell table + accuracy
per backend — measured 2026-07: YOLO 41/49, Tesseract 36/49, CRNN 4/49
correct last digits on the printed bague suffixes):

```bash
python tools/compare_digit_backends.py            # with border stripping
python tools/compare_digit_backends.py --no-strip # effect of the cropping
```

In the app, the expander "🔬 Einzel-Ergebnisse der Erkennungsmodelle" below
the editor shows every model's raw prediction per cell.

`NumericConsensus` majority-votes: all agree → score 100, 2-of-3 → 90 (loser
kept under `alternatives`), all disagree → best-shaped value with score 60,
single voice → 50–60. On Heure/Jour-Mois agreement is decided on the digit
sequence ("14:30" == "1430") and the display value keeps the separator.
The Bague column instead uses sequence reconstruction (see below); all three
backends vote on the anchor cell.

## Cell dictionary contract

After `CellFormatter`, each cell is a dict with:

- `image`: denoised, normalised, inverted uint8 image — used by TrOCR & friends
- `image_raw`: original cell crop (grayscale, non-inverted) — used by HTR-VT,
  Digit and Sexe models that were trained on the unprocessed input
- `erkannt`: recognised text (filled in by whichever recognizer ran)
- `score`: confidence score (-1 if unknown)
- `skip_ocr`: `True` once a cell has a final value and should not be processed
  by further OCR stages
- `verbesserung`: manual correction from the UI

## Modules

- **PdfConverter** — PDF → JPG pages
- **TableRotator** — corrects scan rotation via vertical line detection
- **TatrExtractor** — Microsoft Table-Transformer crops the table region
- **MergedColumnExtractor** / **MergedRowExtractor** — build per-document
  column/row templates via FastLineDetector, then crop cells accordingly
- **DetectColumns** — pytesseract OCR on header cells; tags columns with
  `is_batch_column`, `is_species_column`, `is_sexe_column`, `is_age_column`,
  `is_jour-mois_column`, `is_heure_column`, `is_alle_column`, `is_poids_column`
- **CellDenoiser** — Keras CNN with weighted MSE loss
- **CellFormatter** — normalises and inverts cells for downstream OCR; keeps
  the raw crop under `image_raw`
- **QuotationMarkDetector** — flags "ditto" cells via white-pixel ratio; on
  wide-content columns (Espèce) additionally via ink shape (small centred
  blob), which catches the tintier notations `''`, `//`, `ii` from the corpus
- **HtrVtRecognizer** — vendored HTR-VT (Vision Transformer + CTC) for
  handwritten species names and the age column (tiny vocabulary Fd/Fnd,
  snapped by FuzzyMatchingAge)
- **DigitRecognizer** — multi-strategy reader for the bague (ring number)
  column and all other numeric columns:
  - Tesseract digits-only OCR (psm 7 for the prefix cell, psm 10 for single
    digits) as the primary backend — robust on the printed parts
  - Optional CRNN+CTC Keras model as a fallback for handwritten digits
  - Optional YOLOv8 per-digit detector (`config/digit_yolo.pt`) as an
    independent third voice
  - **Sequence reconstruction** (bague only): once the first cell (handwritten
    prefix + printed suffix) is decoded, the remaining cells are filled
    arithmetically (`start + offset`) and cross-checked against per-cell digit
    OCR. Wrap-around from 9 → 0 is handled correctly because we increment the
    full number, not just the suffix.
  - If the anchor cannot be read, falls back to per-cell single-digit OCR so
    the UI still gets some output to verify.
- **SexeClassifier** — small CNN classifying sexe (m/w/none, trained from
  `M-W-Classification/dataset/`)
- **TrOCR** — `microsoft/trocr-base-stage1` second opinion for Espèce and the
  numeric columns
- **NumericConsensus** — majority voting over the digit/yolo/trocr predictions
  (see "Numeric ensemble" above)
- **FuzzyMatchingBirdNames** — rapidfuzz post-correction against the species
  catalog
- **FuzzyMatchingAge** — CLOSED vocabulary for the age column: only "Fd" and
  "Fnd" exist on these forms. Every cell ends up as one of those, a ditto
  mark, or empty — OCR garbage is blanked, never shown. Other label sets can
  be passed via `FuzzyMatchingAge(allowed_labels=...)`.

## Robustness notes

- All optional recognizers (HTR-VT, DigitRecognizer, SexeClassifier, TrOCR)
  load their models lazily on first use. A missing model file is logged once
  and the module becomes a no-op — the pipeline keeps running and the user
  can still type the value in via the UI.
- Pipeline-level failures in **non-critical** stages (anything after the cell
  extractors) are caught and logged. Set `DOCDIG_TRACEBACK=1` to print the
  full traceback for debugging.
- `TableRotator` falls back to the unrotated image when no vertical lines are
  detected on a page (the previous behaviour silently dropped every page after
  the first failure).

## Nextcloud-Export

Finalised results (original PDF + CSV, side by side under
`/docdig/<scan-name>/`) can be pushed to the Vogelwarte Nextcloud straight
from the app: sidebar → **☁️ Nextcloud** → Benutzername + App-Passwort
eingeben (Nextcloud: Einstellungen → Sicherheit → „Neues App-Passwort
erstellen") → **Verbinden**. Nach erfolgreichem Verbindungstest erscheint
der Button „☁️ In Nextcloud speichern".

Die Zugangsdaten leben **nur in der Browser-Session** — sie werden nie auf
dem Server gespeichert und nicht zwischen Nutzern geteilt. Optional können
`NEXTCLOUD_URL` / `NEXTCLOUD_USER` / `NEXTCLOUD_DIR` in `.env` gesetzt
werden (siehe `.env.example`) — sie füllen dann nur das Login-Formular vor.
Uploads laufen über WebDAV (`libs/nextcloud.py`).

TLS: Die Vogelwarte-Nextcloud nutzt ein intern signiertes Zertifikat. Die
App verifiziert deshalb über den **System-Truststore** (macOS-Schlüsselbund
etc., via `truststore`) — auf verwalteten Geräten funktioniert HTTPS damit
ohne Zutun. Falls die CA fehlt (z. B. im Docker-Container):
`NEXTCLOUD_CA_BUNDLE=/pfad/ca.pem` setzen, oder als letzter Ausweg die
Checkbox „Zertifikatsprüfung deaktivieren" im Login-Formular.

## Deployment (GitLab CI)

`.gitlab-ci.yml` stages — everything past build is **manual**:

```
push auf main/testing
  -> build         Docker-Image (docker/Dockerfile), Tags: SHA, Branch,
                   latest (nur main); Registry registry.vogelwarte.ch
  -> health-check  Container-Start + /_stcore/health (bis zu 20 Versuche)
  -> promote       manuell: taggt latest -> testing bzw. -> production
  -> deploy        manuell: SSH auf $DEPLOY_HOST, docker pull + run
                   testing  -> Port 8502 (Container docdig-testing)
                   production -> Port 8501 (Container docdig-production)
```

Deploy-Ablauf für einen neuen Stand:

1. Commit + push auf `main` → build + health-check laufen automatisch.
2. `main` in `testing` mergen und pushen → build läuft, dann im GitLab-UI
   `promote_to_testing` und `deploy_testing` manuell starten → Test unter
   `http://<host>:8502`.
3. Wenn gut: `testing` in `production` mergen, `promote_to_production` +
   `deploy_production` manuell starten.

Wichtig für den Server (einmalig):

- **HTR-VT-Checkpoint** liegt nicht im Image. Auf dem Host ablegen und den
  `docker run` in `.gitlab-ci.yml` (oder das Compose-Setup) um
  `-v /opt/docdig/htr_vt:/app/config/htr_vt:ro` ergänzen — sonst läuft die
  Espèce-Spalte über TrOCR-Fallback.
- **Nextcloud-Zugang**: `--env-file /opt/docdig/.env` an den `docker run`
  hängen (oder die drei `NEXTCLOUD_*`-Variablen als GitLab-CI-Variablen
  setzen und durchreichen), sonst fehlt der Upload-Button im Deployment.
- Benötigte CI-Variablen (bereits im Projekt konfiguriert, sonst unter
  Settings -> CI/CD -> Variables): `CI_REGISTRY_USER`,
  `CI_REGISTRY_PASSWORD`, `SSH_PRIVATE_KEY` (base64), `DEPLOY_HOST`,
  `DEPLOY_USER`.

Lokal entspricht `docker compose up --build` dem Produktions-Image
(inkl. HTR-Mount + `.env`-Durchreichung, siehe `docker-compose.yml`).

## Vendor code

`src/vendor/htr_vt/` contains a snapshot of
[Intellindust-AI-Lab/HTR-VT](https://github.com/Intellindust-AI-Lab/HTR-VT)
(Pattern Recognition 2025). Only the inference path is used; training-only
files were dropped.
