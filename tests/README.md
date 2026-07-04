# Tests

## Running

```bash
# fast deterministic suite (default)
python3 -m pytest tests/

# include end-to-end runs against the real scan in data/input/
python3 -m pytest tests/ -m integration

# everything
python3 -m pytest tests/ -m ''
```

## What's covered

| File                                  | Marker        | Runs in   | What it locks down |
|---------------------------------------|---------------|-----------|--------------------|
| `test_bague_sequence.py`              | -             | < 1 s     | Anchor parsing, arithmetic extrapolation, wrap-around, manual-edit preservation, zero-padding, score/flag annotations |
| `test_columns.py`                     | -             | < 1 s     | Shared column-flag helpers: numeric/compound classification, label fallback ("Spalte N") so the UI shows every column |
| `test_yolo_digits.py`                 | -             | < 1 s     | YOLO detection→number composition: left-to-right ordering, confidence filtering, cross-class overlap resolution, no-model no-op |
| `test_digit_recognizer.py`            | -             | < 1 s     | DigitRecognizer dispatch (bague sequence + all four numeric column types) with OCR/YOLO backends mocked; YOLO stays a separate voice |
| `test_quotation_mark_detector.py`     | -             | < 1 s     | Ditto detection on synthetic white-vs-inked cells, header + batch-column skip |
| `test_pipeline_robustness.py`         | -             | < 1 s     | Pipeline skips failing non-critical stages, propagates critical failures |
| `test_pipeline_integration.py`        | `integration` | ~60–70 s  | Real PDF (CdB_10) through TATR + recognition; bague sequence after manual anchor; per-cell digit OCR floor |
| `test_files3_corpus.py`               | `integration` | ~3 min    | 6-page labelled corpus (`files-3/`): metadata is parseable, every CSV is a contiguous arithmetic sequence, page-1 anchor-anywhere works end-to-end |

## Fixtures available

| Location                                    | What it is |
|---------------------------------------------|------------|
| `data/input/scan_1972_*.pdf` (symlink)      | The real test scan |
| `data/input/tatr/page_*.jpg` (symlink)      | Cached TATR-cropped pages — speeds up integration runs |
| `config/` (symlinks for TATR weights)       | All required model files: TATR, denoise; HTR-VT auto-discovered under `FelixUpload/inferencedata/digC/` |
| `tests/fixtures/cells/bague/`               | 49 real bague cells from page 0 + `labels.json` with the full ring sequence 144702 → 144750 |
| `files-3/` (symlink)                        | 6-page labelled corpus (`scan_1972_sample.pdf` + 6 `*.expected.csv` + `meta.json`). Bague range 90359–90650, page 1 has 8 rows where no bird was registered, ditto-marks in species column |

## Helper: regenerate the bague fixtures

```bash
python3 tools/extract_bague_fixtures.py            # default: 1 page, ~49 cells
python3 tools/extract_bague_fixtures.py --pages 3  # more pages
```

The script reuses cached TATR pages from `data/input/tatr/` so it runs in
seconds. It preserves any non-empty entries in `labels.json` — re-running
won't clobber labels you've already filled in.

## Why some integration tests are "soft"

`test_bague_per_cell_ocr_accuracy` only asserts ≥ 50 % accuracy on the
printed-digit cells. Real-world Tesseract on table-bordered cells with tiny
fonts tops out around 55–60 % — pixel-perfect numbers come from the **sequence
reconstruction** seeded by a manually-typed anchor, which is what the user
workflow does anyway (`test_bague_sequence_after_manual_anchor` covers that
path, and it asserts 100 % equality with the labels).

`test_consensus.py` additionally locks the 3-voice majority voting for the
numeric columns (digit + yolo + trocr): all-agree → 100, 2-of-3 → 90 with the
loser preserved under `alternatives`, full disagreement → best shape at 60,
and digit-sequence agreement on the compound columns ("14:30" == "1430").

## Testing philosophy (agentic development)

The suite is split so an agent (or a human) can verify changes cheaply:

1. **Deterministic core** (default `pytest` run): every model backend is
   mocked or optional. No network, no model files, no GPU, < 5 s. This is the
   feedback loop for refactors — if it's red, the logic broke, not the data.
2. **Fixture-driven accuracy tests** (auto-skip when fixtures are missing):
   real labelled cells under `tests/fixtures/`; adding fixtures is purely
   additive and can never turn the suite red.
3. **Integration runs** (`-m integration`, opt-in): full pipeline against real
   scans and the `files-3/` labelled corpus with expected CSVs as ground truth.

Rule of thumb: decision logic (voting, sequencing, composition) lives in pure
functions (`libs/yolo_digits.compose_detections`, `libs/bague_sequence`,
`libs/columns`) so it is testable without loading any model.

## What's still missing

To make end-to-end accuracy tests possible, the following would help:

- **Trained `sexe_model.keras`** — would unlock a `test_sexe_classifier_on_real_cells`
  test (analogous to the bague one). Train with `python tools/train_sexe_model.py`
  from `M-W-Classification/dataset/` (classes `m` / `none` / `w`).
- **Labelled numeric-cell fixtures** — a few Aile/Poids/Heure crops with
  ground truth under `tests/fixtures/cells/numeric/` would turn the ensemble
  voting into a measurable accuracy number per backend (tesseract vs. CRNN
  vs. YOLO vs. TrOCR).
- **Hand-labelled species fixtures** — drop a few PNG/label pairs into
  `tests/fixtures/cells/species/` to validate HTR-VT against ground truth.

Everything in `tests/fixtures/` is auto-skipped when missing, so adding
fixtures is purely additive — no test ever turns red because something is
absent.

## Cell dict invariants (locked by the bague tests)

After `DigitRecognizer` and the UI rebuild helpers, every cell in a bague
column has:

- `is_anchor` (bool) — exactly one per column, on the first non-header cell
- `is_extrapolated` (bool) — True when value came from the sequence and was
  *not* confirmed by per-cell OCR; only affects the UI marker
- `is_manual_edit` (bool) — True after `mark_manual_edit()`; `rebuild_batch_sequence`
  refuses to overwrite these unless `overwrite_manual=True` is passed
- `score` (int) — 100 (anchor or OCR-confirmed), 70 (extrapolated), 30
  (per-cell only, no anchor), -1 (empty)
- `skip_ocr` (bool) — downstream OCR stages leave this cell alone
- `erkannt` (str) — the recognised text
