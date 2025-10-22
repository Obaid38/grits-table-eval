# app.py
# Streamlit UI for GriTS (Top / Loc / Con) + Combined
# Reuses your implementations from grits_demo.py
# New: "JSON format required for GriTS" section with schema + dummy examples (+ download)

import json
import streamlit as st
from grits_demo import parse_table_from_json, grits_top, grits_loc, grits_con

# ---------- Page + Styles ----------
st.set_page_config(page_title="GriTS Scorer", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1400px; }
.header-card {
  background: linear-gradient(135deg, rgba(74,144,226,.12), rgba(155,81,224,.12));
  border: 1px solid rgba(180,180,180,.25);
  padding: 22px 24px; border-radius: 18px; margin-bottom: 22px;
}
.card { background: #fff; border: 1px solid rgba(180,180,180,.25); border-radius: 16px;
  padding: 18px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.card h3 { margin: 0 0 6px 0; font-size: 1.05rem; }
.card .desc { font-size: .92rem; color: rgba(60,60,60,.85); margin-bottom: 12px; }
.metric-value { font-weight: 800; font-size: 2.0rem; letter-spacing: -0.02em; margin: 6px 0 2px 0; }
.combined {
  background: linear-gradient(135deg, rgba(46,204,113,.18), rgba(39,174,96,.18));
  border: 1px solid rgba(39,174,96,.35); border-radius: 16px; padding: 16px 18px;
}
.combined .label { font-weight: 700; font-size: 1.0rem; color: #1f5132; }
.combined .value { font-weight: 900; font-size: 2.2rem; letter-spacing: -0.02em; color: #145a32; }
code, pre { font-size: 0.86rem !important; }
footer { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("GriTS Inputs")
    st.caption("Upload **Ground Truth** and **Prediction** JSONs in the required schema. "
               "If you skip, the app will try to load `gt_doc1.json` and `pred_doc1.json` from this folder.")
    gt_file = st.file_uploader("Ground Truth JSON", type=["json"], key="gt")
    pred_file = st.file_uploader("Prediction JSON", type=["json"], key="pred")
    st.divider()
    decimals = st.slider("Display decimals", 2, 6, 4)
    show_details = st.checkbox("Show raw JSON (full)", value=False)

# ---------- Header ----------
st.markdown("<div class='header-card'>", unsafe_allow_html=True)
st.markdown("<h1 class='big-title'>GriTS Scorer</h1>", unsafe_allow_html=True)
st.caption("Evaluate **table-heavy** extraction with three complementary heads: "
           "**Topology (structure)**, **Location**, and **Content**, plus a combined score.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- JSON format section (schema + dummy) ----------
SCHEMA_TEXT = """\
REQUIRED JSON FORMAT (per table)

Top-level fields:
- page_id: string (e.g., "doc1_p0")
- table_id: string (e.g., "t1")
- n_rows: integer >= 1
- n_cols: integer >= 1
- cells: array of objects with:
  - row_start: integer (1-based)
  - row_end  : integer (>= row_start)
  - col_start: integer (1-based)
  - col_end  : integer (>= col_start)
  - bbox     : [left, top, right, bottom] floats in [0,1] or nulls (needed for GriTS_Loc)
  - text     : string (needed for GriTS_Con; empty string allowed)

Notes:
- Indices are 1-based; spans express merged cells.
- bbox uses normalized image coords in [0,1].
- For GriTS_Top only, bbox and text may be null/empty.
- Include blank cells too (use empty text "").
"""

DUMMY_GT = {
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

# Intentional mistakes: header merge; last-number split; tiny bbox shift + text noise
DUMMY_PRED = {
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

with st.expander("📄 JSON format required for GriTS (schema + dummy examples)", expanded=False):
    tabs = st.tabs(["Schema", "Dummy GT", "Dummy Pred"])
    with tabs[0]:
        st.code(SCHEMA_TEXT, language="markdown")
    with tabs[1]:
        gt_text = json.dumps(DUMMY_GT, indent=2, ensure_ascii=False)
        st.code(gt_text, language="json")
        st.download_button("Download dummy GT JSON", gt_text, file_name="gt_doc1.json", mime="application/json")
    with tabs[2]:
        pred_text = json.dumps(DUMMY_PRED, indent=2, ensure_ascii=False)
        st.code(pred_text, language="json")
        st.download_button("Download dummy Pred JSON", pred_text, file_name="pred_doc1.json", mime="application/json")

# ---------- Load inputs (uploads or fallback) ----------
gt_obj = pred_obj = None
if gt_file and pred_file:
    gt_obj = json.load(gt_file)
    pred_obj = json.load(pred_file)
else:
    # fallback to files in working folder
    try:
        with open("gt_doc1.json", "r", encoding="utf-8") as f:
            gt_obj = json.load(f)
        with open("pred_doc1.json", "r", encoding="utf-8") as f:
            pred_obj = json.load(f)
        st.info("Using local fallback: **gt_doc1.json** / **pred_doc1.json**")
    except Exception:
        st.error("Please upload **both** JSON files or place `gt_doc1.json` and `pred_doc1.json` next to `app.py`.")
        st.stop()

# ---------- Score using your functions ----------
gt_tbl = parse_table_from_json(gt_obj)
pr_tbl = parse_table_from_json(pred_obj)

top = grits_top(gt_tbl, pr_tbl)
loc = grits_loc(gt_tbl, pr_tbl)
con = grits_con(gt_tbl, pr_tbl)
combined = (top + loc + con) / 3.0

# ---------- Combined banner ----------
st.markdown("<div class='combined'>", unsafe_allow_html=True)
st.markdown(f"<div class='label'>Combined GriTS</div><div class='value'>{combined:.{decimals}f}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# ---------- Three metric cards ----------
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>GriTS_Top</h3>", unsafe_allow_html=True)
    st.markdown("<div class='desc'><b>Topology-only.</b> Compares the grid structure (rows, columns, spans). Partial credit for splits/merges.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{top:.{decimals}f}</div>", unsafe_allow_html=True)
    st.progress(min(max(top, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>GriTS_Loc</h3>", unsafe_allow_html=True)
    st.markdown("<div class='desc'><b>Location-aware.</b> Uses per-cell boxes; rewards tight coverage and penalizes leakage.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{loc:.{decimals}f}</div>", unsafe_allow_html=True)
    st.progress(min(max(loc, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>GriTS_Con</h3>", unsafe_allow_html=True)
    st.markdown("<div class='desc'><b>Content-aware.</b> Token-level match of cell text inside the aligned grid.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{con:.{decimals}f}</div>", unsafe_allow_html=True)
    st.progress(min(max(con, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Full raw JSON (non-collapsed) ----------
if show_details:
    st.markdown("---")
    st.subheader("Raw JSON (full)")
    gt_text_user = json.dumps(gt_obj, indent=2, ensure_ascii=False)
    pr_text_user = json.dumps(pred_obj, indent=2, ensure_ascii=False)
    left, right = st.columns(2)
    with left:
        st.caption("Ground Truth JSON")
        st.code(gt_text_user, language="json")
        st.download_button("Download GT JSON", gt_text_user, file_name="gt.json", mime="application/json")
    with right:
        st.caption("Prediction JSON")
        st.code(pr_text_user, language="json")
        st.download_button("Download Pred JSON", pr_text_user, file_name="pred.json", mime="application/json")
