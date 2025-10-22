

import json, re, sys, glob, csv, os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set
from collections import Counter

@dataclass
class Cell:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    bbox: Tuple[float, float, float, float]  # (left, top, right, bottom) or (None,..)
    text: str

@dataclass
class Table:
    page_id: str
    table_id: str
    n_rows: int
    n_cols: int
    cells: List[Cell]

def normalize_text(s: str) -> str:
    import re
    s2 = s.lower()
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def tokenize(s: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", s.lower())

def parse_table_from_json(obj: Dict[str, Any]) -> Table:
    cells = []
    for c in obj["cells"]:
        bbox_raw = c.get("bbox")
        if bbox_raw is None:
            bbox = (None, None, None, None)
        else:
            bbox = tuple(x if x is None else float(x) for x in bbox_raw)
        cells.append(Cell(
            row_start=int(c["row_start"]),
            row_end=int(c["row_end"]),
            col_start=int(c["col_start"]),
            col_end=int(c["col_end"]),
            bbox=bbox,
            text=c.get("text","")
        ))
    return Table(
        page_id=obj.get("page_id","doc_p0"),
        table_id=obj.get("table_id","t1"),
        n_rows=int(obj["n_rows"]),
        n_cols=int(obj["n_cols"]),
        cells=cells
    )

def unit_positions(cell: Cell) -> Set[Tuple[int,int]]:
    return {(r,c) for r in range(cell.row_start, cell.row_end+1)
                  for c in range(cell.col_start, cell.col_end+1)}

def spans_intersect(a: Cell, b: Cell) -> bool:
    return not (a.row_end < b.row_start or b.row_end < a.row_start or
                a.col_end < b.col_start or b.col_end < a.col_start)

def bbox_area(b: Tuple[float,float,float,float]) -> float:
    l,t,r,bottom = b
    if l is None or t is None: 
        return 0.0
    return max(0.0, r-l) * max(0.0, bottom-t)

def overlap_with_union_area(gt_bbox, pred_bboxes, samples: int = 150):
    """Sampling-based overlap with union area."""
    if gt_bbox[0] is None:
        return 0.0, 0.0, 0.0
    if not pred_bboxes:
        return bbox_area(gt_bbox), 0.0, 0.0
    dx = 1.0 / samples
    dy = 1.0 / samples
    def inside(b, x, y):
        l,t,r,bottom = b
        return (x >= l) and (x <= r) and (y >= t) and (y <= bottom)
    union_count = inter_count = gt_count = 0
    for i in range(samples):
        y = (i + 0.5)*dy
        for j in range(samples):
            x = (j + 0.5)*dx
            in_gt = inside(gt_bbox, x, y)
            in_union = any(inside(pb, x, y) for pb in pred_bboxes)
            if in_gt: gt_count += 1
            if in_union: union_count += 1
            if in_gt and in_union: inter_count += 1
    A_cell = dx*dy
    return gt_count*A_cell, union_count*A_cell, inter_count*A_cell

def f1(p: float, r: float) -> float:
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

def grits_top(gt: Table, pred: Table) -> float:
    gt_positions = {id(c): unit_positions(c) for c in gt.cells}
    pred_positions = {id(c): unit_positions(c) for c in pred.cells}
    scores = []
    for g in gt.cells:
        U = gt_positions[id(g)]
        cand = [p for p in pred.cells if spans_intersect(g,p)]
        if not cand:
            scores.append(0.0); continue
        V = set().union(*(pred_positions[id(p)] for p in cand))
        overlap = U & V
        rec = len(overlap)/len(U) if U else 0.0
        prec = len(overlap)/len(V) if V else 0.0
        scores.append(f1(prec, rec))
    return sum(scores)/len(scores) if scores else 0.0

def grits_loc(gt: Table, pred: Table) -> float:
    scores = []
    for g in gt.cells:
        if g.bbox[0] is None: continue
        cand = [p for p in pred.cells if spans_intersect(g,p) and p.bbox[0] is not None]
        if not cand:
            scores.append(0.0); continue
        A_gt, A_union, A_inter = overlap_with_union_area(g.bbox, [p.bbox for p in cand], samples=140)
        rec = 0.0 if A_gt==0 else (A_inter / A_gt)
        prec = 0.0 if A_union==0 else (A_inter / A_union)
        scores.append(f1(prec, rec))
    return sum(scores)/len(scores) if scores else 0.0

def grits_con(gt: Table, pred: Table) -> float:
    scores = []
    for g in gt.cells:
        gt_tokens = tokenize(normalize_text(g.text))
        cand = [p for p in pred.cells if spans_intersect(g,p)]
        pred_text = " ".join(normalize_text(p.text) for p in cand)
        pred_tokens = tokenize(pred_text)
        if not gt_tokens and not pred_tokens:
            scores.append(1.0); continue
        c_gt = Counter(gt_tokens); c_pr = Counter(pred_tokens)
        common = sum((c_gt & c_pr).values())
        prec = 0.0 if sum(c_pr.values())==0 else common / sum(c_pr.values())
        rec  = 0.0 if sum(c_gt.values())==0 else common / sum(c_gt.values())
        scores.append(f1(prec, rec))
    return sum(scores)/len(scores) if scores else 0.0

def run_single(gt_path="gt_doc1.json", pred_path="pred_doc1.json", doc_label="Doc1", write_csv=True):
    gt = parse_table_from_json(json.load(open(gt_path, "r", encoding="utf-8")))
    pr = parse_table_from_json(json.load(open(pred_path, "r", encoding="utf-8")))
    top = grits_top(gt, pr)
    loc = grits_loc(gt, pr)
    con = grits_con(gt, pr)
    overall = (top + loc + con) / 3.0
    print(f"{doc_label} — GriTS scores")
    print(f"  GriTS_Top (overall Table Structure): {top:.4f}")
    print(f"  GriTS_Loc (location-aware): {loc:.4f}")
    print(f"  GriTS_Con (content-aware): {con:.4f}")
    print(f"  Combined : {overall:.4f}")
    # if write_csv:
    #     out_csv = os.path.join(os.path.dirname(os.path.abspath(gt_path)), "grits_summary.csv")
    #     header = not os.path.exists(out_csv)
    #     with open(out_csv, "a", newline="", encoding="utf-8") as f:
    #         w = csv.DictWriter(f, fieldnames=["doc","GriTS_Top","GriTS_Loc","GriTS_Con","Combined"])
    #         if header: w.writeheader()
    #         w.writerow({"doc": doc_label, "GriTS_Top": top, "GriTS_Loc": loc, "GriTS_Con": con, "Combined": overall})
    #     print(f"[saved] {out_csv}")

def run_batch(data_dir="."):
    print(f"[scan] looking for gt_*.json / pred_*.json in: {os.path.abspath(data_dir)}")
    gt_files = sorted(glob.glob(os.path.join(data_dir, "gt_*.json")))
    if not gt_files:
        print("[warn] no gt_*.json files found"); return
    rows = []
    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        doc_id = base[3:-5]  # strip 'gt_' and '.json'
        pred_path = os.path.join(data_dir, f"pred_{doc_id}.json")
        if not os.path.exists(pred_path):
            print(f"[skip] missing pred for {doc_id} → expected {pred_path}")
            continue
        gt = parse_table_from_json(json.load(open(gt_path, encoding="utf-8")))
        pr = parse_table_from_json(json.load(open(pred_path, encoding="utf-8")))
        top = grits_top(gt, pr); loc = grits_loc(gt, pr); con = grits_con(gt, pr)
        overall = (top + loc + con) / 3.0
        rows.append({"doc": doc_id, "GriTS_Top": top, "GriTS_Loc": loc, "GriTS_Con": con, "Combined": overall})
        print(f"{doc_id}: top={top:.4f} loc={loc:.4f} con={con:.4f} all={overall:.4f}")
    out_csv = os.path.join(data_dir, "grits_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc","GriTS_Top","GriTS_Loc","GriTS_Con","Combined"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[saved] {out_csv}")

if __name__ == "__main__":
    # CLI
    if len(sys.argv) == 1:
        # default single pair in current folder
        run_single("gt_doc1.json", "pred_doc1.json", doc_label="Doc1", write_csv=True)
    elif len(sys.argv) == 2 and sys.argv[1] == "--batch":
        run_batch(".")
    elif len(sys.argv) == 3 and sys.argv[1] == "--batch":
        run_batch(sys.argv[2])
    elif len(sys.argv) == 3:
        run_single(sys.argv[1], sys.argv[2], doc_label=os.path.splitext(os.path.basename(sys.argv[1]))[0], write_csv=True)
    else:
        print("Usage:")
        print("  python grits_demo.py")
        print("  python grits_demo.py gt_doc.json pred_doc.json")
        print("  python grits_demo.py --batch")
        print("  python grits_demo.py --batch path/to/folder")
