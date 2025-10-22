# Table Extraction Benchmarks with GriTS

Evaluate table-heavy documents using **GriTS**, a grid-based similarity metric that scores how well a predicted table matches ground truth across three heads:

- **GriTS_Top** — *Topology (structure)*: rows, columns, and cell spans (merges/splits).
- **GriTS_Loc** — *Location-aware*: per-cell bounding boxes (geometry/placement).
- **GriTS_Con** — *Content-aware*: per-cell text (token-level match).

This repo includes a minimal scorer (`grits_demo.py`) and an optional Streamlit UI (`app.py`) for quick visualization.

---

## Quick Start

**Single pair (defaults to `gt_doc1.json` / `pred_doc1.json`):**
```bash
python grits_demo.py
Explicit files:

bash
Copy code
python grits_demo.py path/to/gt.json path/to/pred.json
Batch (scan current folder for gt_*.json / pred_*.json):

bash
Copy code
python grits_demo.py --batch
Writes grits_summary.csv with per-document scores.

Streamlit UI (optional):

bash
Copy code
pip install streamlit
streamlit run app.py
Repo Layout
grits_demo.py — scorer + batch runner (computes Top/Loc/Con + Combined)

app.py — Streamlit UI (upload JSONs or use local fallbacks)

gt_doc1.json / pred_doc1.json — small dummy examples

JSON Format Required for GriTS (per table)
Each table (both GT and Pred) must be described by a single JSON object:

txt
Copy code
page_id   : string (e.g., "doc1_p0")
table_id  : string (e.g., "t1")
n_rows    : integer >= 1
n_cols    : integer >= 1
cells     : array of objects:
  - row_start : integer (1-based)
  - row_end   : integer (>= row_start)
  - col_start : integer (1-based)
  - col_end   : integer (>= col_start)
  - bbox      : [left, top, right, bottom] floats in [0,1]  |  nulls (Top-only)
  - text      : string (Content head)                        |  "" allowed
Notes

Indices are 1-based; row_end/col_end enable merged cells (spans).

bbox values are normalized image coordinates in [0,1].
Used by GriTS_Loc. If you don’t have per-cell boxes, set bbox: null.

text is used by GriTS_Con. Use normalized text (trim/collapse spaces) on both GT and Pred.

Minimal Dummy Examples
Ground Truth (gt_doc1.json)

json
Copy code
{
  "page_id": "doc1_p0",
  "table_id": "t1",
  "n_rows": 3,
  "n_cols": 3,
  "cells": [
    {"row_start":1,"row_end":1,"col_start":1,"col_end":1,"bbox":[0.10,0.10,0.30,0.18],"text":"Type"},
    {"row_start":1,"row_end":1,"col_start":2,"col_end":2,"bbox":[0.30,0.10,0.60,0.18],"text":"Instrument"},
    {"row_start":1,"row_end":1,"col_start":3,"col_end":3,"bbox":[0.60,0.10,0.90,0.18],"text":"Number"},

    {"row_start":2,"row_end":2,"col_start":1,"col_end":1,"bbox":[0.10,0.18,0.30,0.26],"text":"Partial Payment"},
    {"row_start":2,"row_end":2,"col_start":2,"col_end":2,"bbox":[0.30,0.18,0.60,0.26],"text":"Payorder"},
    {"row_start":2,"row_end":2,"col_start":3,"col_end":3,"bbox":[0.60,0.18,0.90,0.26],"text":"12198881"},

    {"row_start":3,"row_end":3,"col_start":1,"col_end":1,"bbox":[0.10,0.26,0.30,0.34],"text":"Advance"},
    {"row_start":3,"row_end":3,"col_start":2,"col_end":2,"bbox":[0.30,0.26,0.60,0.34],"text":"Cheque"},
    {"row_start":3,"row_end":3,"col_start":3,"col_end":3,"bbox":[0.60,0.26,0.90,0.34],"text":"998877"}
  ]
}
Prediction (pred_doc1.json) — with intentional errors

json
Copy code
{
  "page_id": "doc1_p0",
  "table_id": "t1",
  "n_rows": 3,
  "n_cols": 3,
  "cells": [
    {"row_start":1,"row_end":1,"col_start":1,"col_end":1,"bbox":[0.10,0.10,0.30,0.185],"text":"Type"},
    {"row_start":1,"row_end":1,"col_start":2,"col_end":3,"bbox":[0.31,0.10,0.90,0.185],"text":"Instrument Number"},

    {"row_start":2,"row_end":2,"col_start":1,"col_end":1,"bbox":[0.10,0.18,0.30,0.26],"text":"Partial  Payment"},
    {"row_start":2,"row_end":2,"col_start":2,"col_end":2,"bbox":[0.30,0.18,0.60,0.26],"text":"Payorder"},
    {"row_start":2,"row_end":2,"col_start":3,"col_end":3,"bbox":[0.60,0.18,0.90,0.26],"text":"12198881"},

    {"row_start":3,"row_end":3,"col_start":1,"col_end":1,"bbox":[0.10,0.26,0.30,0.34],"text":"Advance"},
    {"row_start":3,"row_end":3,"col_start":2,"col_end":2,"bbox":[0.30,0.26,0.60,0.34],"text":"Cheque"},
    {"row_start":3,"row_end":3,"col_start":3,"col_end":3,"bbox":[0.60,0.26,0.75,0.34],"text":"998"},
    {"row_start":3,"row_end":3,"col_start":3,"col_end":3,"bbox":[0.75,0.26,0.90,0.34],"text":"877"}
  ]
}
Running the Metric
CLI: use grits_demo.py to compute the three heads and a simple Combined average.

UI: app.py shows the three scores as cards and a combined banner; upload JSONs or use local fallbacks.

Why This JSON Format?
GriTS scores require comparable structure between GT and Pred. The format enforces:

Topology (GriTS_Top) — Each cell’s row/column span captures merges/splits (e.g., a header spanning two columns).

Location (GriTS_Loc) — Per-cell bounding boxes (normalized to [0,1]) let us evaluate geometric overlap and penalize leakage/misalignment.

Content (GriTS_Con) — Cell text is compared in the aligned grid regions, rewarding correct placement and content.

By standardizing on this schema, we can plug in outputs from different systems (OCR, LLM/VLMs, vendor APIs) and evaluate them consistently with a single, defensible metric.

