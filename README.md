# grits-table-eval

> Evaluate how well a model extracted a table from a document — not just the text, but the structure, geometry, and content together.

Standard OCR metrics like character error rate will tell you that 98% of the characters came through. They won't tell you that the model merged two columns, shifted a number into the wrong row, or invented a "Branch Code" column that was never in the original document. Those failures only show up when you evaluate the table as a grid.

This project implements the [GriTS metric](https://arxiv.org/abs/2203.12555) (Grid Table Similarity, Landing AI 2022) — a three-headed evaluation framework that scores table extraction at the structural, spatial, and textual level simultaneously.

---

## 🔍 Why Tables Break OCR

Tables in scanned documents are structurally adversarial for most extraction pipelines:

- Merged cells span multiple rows or columns — a line-based detector treats each span as a separate region
- Borderless tables have no visual separator between cells — a column detector can't find columns it can't see
- Multi-line cell content breaks row-counting heuristics
- Dense financial tables (bank statements, invoices) combine all of the above in a single document

The result is that a pipeline can achieve high character-level accuracy while the structural output is completely wrong. A number gets extracted but placed in the wrong row. A header gets split across two cells. An extra column appears out of nowhere. None of these failures show up in CER. GriTS was designed precisely to catch them.

---

## 📐 The Three Scoring Heads

**GriTS_Top — Topology**

- Scores the grid structure: which cells exist, how they span, which are merged or split
- Each GT cell is matched against all prediction cells that overlap its span
- Score is an F1 over the set of unit grid positions each cell occupies
- A column the model split in two loses partial credit, not zero — penalisation is proportional to the structural error

**GriTS_Loc — Location**

- Extends topology scoring with spatial geometry
- Each cell's bounding box (normalised to image dimensions, range 0–1) is compared via intersection-over-union
- Uses sampling-based overlap calculation — no OpenCV dependency
- Catches cases where the structure is right but the detector placed boxes in the wrong location

**GriTS_Con — Content**

- Adds token-level text comparison to the aligned grid regions
- Text is lowercased, whitespace-collapsed, then tokenised on `[a-z0-9]+`
- F1 of matched tokens between GT and prediction — partial credit for partial matches
- Catches OCR garbling, truncated strings, and content that landed in the wrong cell

**Combined** is the unweighted mean of all three. The individual heads are the diagnostic signal — a high Top but low Con gap points to an OCR problem downstream; a low Loc gap with good Top points to a box regression problem, not a recognition problem.

---

## 🏗️ Architecture

```
grits_demo.py
    Cell (dataclass)           row/col span + bbox + text
    Table (dataclass)          page_id, table_id, dimensions, cell list
    normalize_text()           lowercase + whitespace collapse
    tokenize()                 extract alphanumeric tokens via regex
    parse_table_from_json()    JSON -> Table
    unit_positions()           set of (row, col) positions a cell occupies
    spans_intersect()          overlap check for two cell spans
    bbox_area()                bounding box area calculation
    overlap_with_union_area()  sampling-based IoU (150x150 grid default)
    f1()                       precision/recall -> F1
    grits_top()                topology head
    grits_loc()                location head
    grits_con()                content head
    run_single()               score one GT/Pred pair, print to stdout
    run_batch()                scan directory for gt_*/pred_* pairs, write CSV

app.py
    Streamlit UI
    - sidebar file upload (GT + Pred JSONs)
    - falls back to local gt_doc1.json / pred_doc1.json
    - three metric cards (GriTS_Top, GriTS_Loc, GriTS_Con)
    - combined score banner
    - configurable decimal precision
    - raw JSON viewer
    - inline schema reference with downloadable dummy examples
```

No deep learning, no GPU, no model weights. The scorer is pure Python and can be dropped into any evaluation pipeline as a module.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Scoring engine | Pure Python — `dataclasses`, `collections.Counter`, `re` |
| Geometric overlap | Sampling-based (no OpenCV or shapely dependency) |
| Web UI | Streamlit |
| Data format | JSON per-table schema |
| Dependencies | `streamlit`, `pandas`, `numpy` |

---

## 📁 Project Structure

```
grits-table-eval/
|- grits_demo.py        core scoring engine — metrics, CLI runner, batch mode
|- app.py               Streamlit web UI
|- gt_doc1.json         demo ground truth — simple 3x3 payment table
|- pred_doc1.json       demo prediction — header merge + number split errors
|- gt_doc2.json         demo ground truth — 8x7 bank statement
|- pred_doc2.json       demo prediction — hallucinated column, OCR typos, balance errors
|- requirements.txt
|- README.md
```

---

## ⚡ Setup

```bash
git clone https://github.com/Obaid38/grits-table-eval.git
cd grits-table-eval
pip install -r requirements.txt
```

Python 3.9+. No virtual environment strictly required for a quick run.

---

## 🚀 Usage

**Default — scores `gt_doc1.json` vs `pred_doc1.json` in the working directory:**

```bash
python grits_demo.py
```

**Explicit file pair:**

```bash
python grits_demo.py path/to/gt.json path/to/pred.json
```

**Batch mode — scans a directory for all `gt_*.json` / `pred_*.json` pairs:**

```bash
python grits_demo.py --batch
python grits_demo.py --batch path/to/folder
```

Batch mode writes `grits_summary.csv` with per-document scores.

**Streamlit UI:**

```bash
streamlit run app.py
```

Upload your GT and Pred JSONs in the sidebar, or let the app fall back to the local demo files. The UI renders score cards, a combined banner, progress bars, and an inline schema reference with downloadable example files.

---

## 📋 JSON Schema

Each table — ground truth or prediction — is a single JSON object:

```
{
  "page_id":  string,         e.g. "invoice_p0"
  "table_id": string,         e.g. "t1"
  "n_rows":   integer >= 1,
  "n_cols":   integer >= 1,
  "cells": [
    {
      "row_start": integer,   1-based
      "row_end":   integer,   >= row_start  (== row_start for a non-merged cell)
      "col_start": integer,   1-based
      "col_end":   integer,   >= col_start
      "bbox":      [left, top, right, bottom]  floats in [0.0, 1.0]  |  null
      "text":      string     (empty string for blank cells, not null)
    },
    ...
  ]
}
```

Notes:
- Indices are 1-based. A cell spanning two columns has `col_start=2, col_end=3`.
- `bbox` uses normalised image coordinates. Required for GriTS_Loc — set to `null` if unavailable (GriTS_Loc scores those cells as 0).
- Include every cell, including headers and visually blank cells.
- The schema is intentionally minimal — it carries only what each scoring head needs.

---

## 🧪 Demo Data

**Doc1 — Payment table (3×3):**
A simple header + two-row table with payment type, instrument, and reference number.
The prediction merges columns 2–3 into one cell and splits the last number into two adjacent fragments.
This isolates the effect of column merges on GriTS_Top without any location or content errors.

**Doc2 — Bank statement (8×7 GT vs 8×8 prediction):**
A multi-row financial transaction table. The prediction model:
- Hallucinates a "Branch Code" column (8 columns instead of 7)
- Drops a zero from a salary figure (350,000 becomes 35,000)
- Introduces OCR typos throughout transaction descriptions
- Misspells "Balance" as "Ballance" in the header
- Computes the wrong closing balance as a result

This example shows all three heads degrading simultaneously — the failure mode you see on real dense documents from bank statements, invoices, and financial reports.

---

## 📚 Research Background

GriTS was introduced in [*GriTS: Grid Table Similarity Metric for Table Structure Recognition*](https://arxiv.org/abs/2203.12555) (Smock, Pesala, Abraham — Landing AI, 2022), developed alongside the [TableTransformer](https://github.com/microsoft/table-transformer) model.

The core problem GriTS addresses is that table structure is a two-dimensional relational object. A flat sequence metric like BLEU or CER has no concept of rows, columns, or cell spans. A prediction that gets every character correct but shifts a row downward scores perfectly on CER and fails completely on any downstream database query or form extraction. GriTS penalises exactly that class of failure.

The three-head design separates structural, spatial, and textual correctness so you can diagnose *where* a pipeline breaks rather than receiving a single aggregate score that conflates unrelated failure modes.

---

## ⚠️ Limitations

- Location scoring uses a 150×150 sampling grid rather than exact polygon intersection. Accurate for rectangular bboxes; increase the `num_samples` parameter in `grits_demo.py` for higher precision.
- One table object per JSON file. Multi-table documents require one file per table.
- The combined score is unweighted. For text-only pipelines that don't produce bounding boxes, consider ignoring GriTS_Loc or setting `bbox` to `null` uniformly.

---

## 📄 License

MIT
