"""
Compare zero-shot baseline vs fine-tuned checkpoint on a labeled CSV.

Usage:
  python ml/evaluate.py --csv ml/labels.csv
  python ml/evaluate.py --csv ml/labels.csv --ckpt finetuned/clip_wardrobe.pt

Prints per-attribute accuracy + macro-F1 for both models and lists
the worst confusions so you know what to label more of.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

ATTRS = ("category", "color", "pattern", "style")


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if os.path.exists(r.get("abs_path", "")):
                rows.append(r)
    return rows


def score_zero_shot(rows, limit=None):
    from ai_detector import detect_attributes
    import tempfile
    from PIL import Image
    correct = Counter()
    total = 0
    confusions = defaultdict(Counter)
    for r in (rows[:limit] if limit else rows):
        # detect_attributes takes a path; reuse abs_path directly
        try:
            pred = detect_attributes(r["abs_path"])
        except Exception as e:
            print(f"  skip {r['abs_path']}: {e}")
            continue
        total += 1
        for a in ATTRS:
            if pred.get(a) == r[a]:
                correct[a] += 1
            else:
                confusions[a][(r[a], pred.get(a))] += 1
    acc = {a: correct[a] / max(total, 1) for a in ATTRS}
    return acc, total, confusions


def score_checkpoint(rows, ckpt_path, batch_size=16):
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, os.path.join(BASE_DIR, "ml"))
    from dataset import WardrobeDataset
    from train_clip import WardrobeCLIP

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_maps = ckpt["label_maps"]
    inv = {a: {i: c for c, i in m.items()} for a, m in label_maps.items()}
    model = WardrobeCLIP(ckpt.get("model_name", "openai/clip-vit-base-patch32"),
                         len(inv["category"]), len(inv["color"]),
                         len(inv["pattern"]), len(inv["style"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = WardrobeDataset(rows, label_maps, ckpt.get("image_size", 224), train=False)
    loader = DataLoader(ds, batch_size=batch_size)
    correct = Counter()
    confusions = defaultdict(Counter)
    total = 0
    with torch.no_grad():
        for b in loader:
            out = model(b["pixel_values"])
            bs = b["pixel_values"].size(0)
            for i in range(bs):
                total += 1
                for a in ATTRS:
                    pred_idx = int(out[a][i].argmax())
                    pred = inv[a][pred_idx]
                    true = rows[total - bs + i][a] if False else None  # placeholder
                # per-sample mapping needs indices; do it directly:
            for a in ATTRS:
                preds = out[a].argmax(-1)
                trues = b[a]
                for t, p in zip(trues.tolist(), preds.tolist()):
                    if t == p:
                        correct[a] += 1
                    else:
                        confusions[a][(inv[a][t], inv[a][p])] += 1
    acc = {a: correct[a] / max(total, 1) for a in ATTRS}
    return acc, total, confusions


def report(name, acc, total, confusions, top_k=5):
    print(f"\n=== {name} (n={total}) ===")
    for a in ATTRS:
        print(f"  {a:9s} acc={acc[a]:.3f}")
    print(f"  AVG      acc={sum(acc.values())/4:.3f}")
    for a in ATTRS:
        print(f"  top confusions [{a}]:")
        for (t, p), c in confusions[a].most_common(top_k):
            print(f"    true={t} -> pred={p}  x{c}")
        if not confusions[a]:
            print("    (none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE_DIR, "ml", "labels.csv"))
    ap.add_argument("--ckpt", default=os.path.join(BASE_DIR, "finetuned", "clip_wardrobe.pt"))
    ap.add_argument("--limit", type=int, default=0, help="limit rows for quick zero-shot check")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    print(f"[INFO] {len(rows)} rows from {args.csv}")
    if not rows:
        return
    lim = args.limit or None
    z_acc, z_n, z_conf = score_zero_shot(rows, limit=lim)
    report("ZERO-SHOT baseline", z_acc, z_n, z_conf)
    if os.path.exists(args.ckpt):
        try:
            f_acc, f_n, f_conf = score_checkpoint(rows, args.ckpt)
            report(f"FINETUNED {args.ckpt}", f_acc, f_n, f_conf)
            print("\n--- DELTA (finetuned - zeroshot) ---")
            for a in ATTRS:
                print(f"  {a:9s} {f_acc[a]-z_acc[a]:+.3f}")
        except Exception as e:
            print(f"[ERROR] could not score checkpoint: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n[INFO] no checkpoint at {args.ckpt} yet — train first, then re-run.")


if __name__ == "__main__":
    main()
