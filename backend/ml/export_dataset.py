"""
Export wardrobe.db + uploads/ into a clean training CSV.

Label model (two taxonomies merged):
  subcategory (Gemini pipeline: shirt/t-shirt/pant/jeans/...) wins when valid,
  else the CLIP category. Exported as `subcat` — the single head target.

Senior-trainer rules:
  1. Trusted rows (PATCH-verified, conf>=0.99) train at full weight.
  2. High-conf pseudo-labels (>= --min-conf) train at reduced weight
     (self-training), NEVER validate.
  3. Photo-verified singletons (--include-ids) join training at full weight.

Usage:
    python ml/export_dataset.py --trusted-only --out ml/labels_trusted.csv
    python ml/export_dataset.py --min-conf 0.8 --out ml/labels_aux.csv
    python ml/export_dataset.py --min-conf 0.8 --include-ids 22,23,39 --out ml/labels_aux.csv
"""
import argparse
import csv
import os
import sqlite3
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, "wardrobe.db")
DEFAULT_OUT = os.path.join(BASE_DIR, "ml", "labels.csv")

sys.path.insert(0, BASE_DIR)
try:
    from ai_detector import CATEGORIES, COLORS, PATTERNS, STYLES
except ImportError:
    CATEGORIES, COLORS, PATTERNS, STYLES = [], [], [], []


def _cols(cur):
    return [r[1] for r in cur.execute("PRAGMA table_info(wardrobe_items)").fetchall()]


def export(trusted_only, min_conf, include_ids, out_path):
    if not os.path.exists(DB_FILE):
        print(f"[ERROR] DB not found: {DB_FILE}")
        sys.exit(1)
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    has_subcat = "subcategory" in _cols(cur)
    sub_sel = "subcategory" if has_subcat else "category"
    if trusted_only:
        rows = cur.execute(
            f"SELECT id,image_path,category,{sub_sel},color,pattern,style,confidence,needs_review"
            " FROM wardrobe_items WHERE needs_review=0 AND confidence>=0.99"
        ).fetchall()
    else:
        rows = cur.execute(
            f"SELECT id,image_path,category,{sub_sel},color,pattern,style,confidence,needs_review"
            " FROM wardrobe_items"
        ).fetchall()
    con.close()

    include_ids = {int(x) for x in include_ids.split(",") if x.strip().isdigit()}
    valid, missing, bad_label, lowconf = [], [], [], 0
    allowed = {
        "color": set(COLORS),
        "pattern": set(PATTERNS),
        "style": set(STYLES),
    }
    for _id, rel, cat, sub, col, pat, sty, conf, nr in rows:
        conf = float(conf or 0.0)
        # unified head target: subcategory when valid, else CLIP category
        subcat = sub if sub in CATEGORIES else (cat if cat in CATEGORIES else None)
        if _id in include_ids:
            pass  # photo-verified: in regardless of confidence
        elif trusted_only:
            pass  # already filtered by SQL
        elif conf < min_conf:
            lowconf += 1
            continue
        abs_p = os.path.join(BASE_DIR, (rel or "").replace("/", os.sep))
        if not rel or not os.path.exists(abs_p):
            missing.append((_id, rel))
            continue
        bad = []
        if subcat is None:
            bad.append(f"subcategory={sub}/{cat}")
        if allowed["color"] and col not in allowed["color"]:
            bad.append(f"color={col}")
        if allowed["pattern"] and pat not in allowed["pattern"]:
            bad.append(f"pattern={pat}")
        if allowed["style"] and sty not in allowed["style"]:
            bad.append(f"style={sty}")
        if bad:
            bad_label.append((_id, ";".join(bad)))
            continue
        valid.append((_id, abs_p, rel, subcat, col, pat, sty, conf, nr))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "abs_path", "rel_path", "category", "color", "pattern", "style", "confidence", "needs_review"])
        w.writerows(valid)

    print(f"[OK] wrote {len(valid)} rows -> {out_path}")
    print(f"     skipped missing-file: {len(missing)}, bad-label: {len(bad_label)}, below-min-conf: {lowconf}")
    if bad_label[:8]:
        print("     e.g. bad label:", bad_label[:8])

    for name, idx in [("category", 3), ("color", 4), ("pattern", 5), ("style", 6)]:
        c = Counter(r[idx] for r in valid)
        print(f"\n-- {name} ({len(c)} classes) --")
        for k, v in c.most_common():
            flag = "  <-- RARE (<5)" if v < 5 else ""
            print(f"   {k:12s} {v:3d}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trusted-only", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--include-ids", default="")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    export(args.trusted_only, args.min_conf, args.include_ids, args.out)
