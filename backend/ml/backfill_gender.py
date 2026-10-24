"""
Backfill gender labels for wardrobe items that predate gender detection.

- Dresses -> female (by construction).
- Everything else -> CLIP menswear/womenswear/unisex zero-shot.
- Weak winners (conf < --min-conf) stay "unisex" so items remain visible
  under both Male and Female filters instead of being hidden wrongly.

Usage (from backend/ with venv active):
    python ml/backfill_gender.py --dry-run     # preview only
    python ml/backfill_gender.py               # write changes
    python ml/backfill_gender.py --force       # also re-label non-unisex rows
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database import SessionLocal
from models import WardrobeItem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.60)
    args = ap.parse_args()

    # Import lazily (loads torch/transformers only when actually running).
    from ai_detector import detect_attributes, GENDERS

    db = SessionLocal()
    q = db.query(WardrobeItem).order_by(WardrobeItem.id)
    if not args.force:
        q = q.filter((WardrobeItem.gender == "unisex") | (WardrobeItem.gender.is_(None)))
    items = q.all()
    print(f"[INFO] {len(items)} items to process (force={args.force})")
    if not items:
        return

    from collections import Counter
    decided = Counter()
    for it in items:
        rel = (it.image_path or "").replace("/", os.sep)
        abs_p = os.path.join(BASE_DIR, rel)
        if not os.path.exists(abs_p):
            print(f"  skip id={it.id}: missing file {rel}")
            continue
        try:
            attrs = detect_attributes(abs_p)
        except Exception as e:
            print(f"  skip id={it.id}: detection failed: {e}")
            continue
        gender = attrs.get("gender", "unisex") or "unisex"
        if gender not in GENDERS:
            gender = "unisex"
        gconf = float(attrs.get("gender_confidence", 0.0) or 0.0)
        if gconf < args.min_conf and it.category != "dress":
            gender = "unisex"
        decided[gender] += 1
        mark = "DRY" if args.dry_run else "SET"
        print(f"  [{mark}] id={it.id:3d} {it.category:8s} {it.color:8s} -> {gender:7s} (gconf={gconf:.2f})")
        if not args.dry_run:
            it.gender = gender
    if not args.dry_run:
        db.commit()
        print("[OK] committed.")
    print("decided:", dict(decided))


if __name__ == "__main__":
    main()
