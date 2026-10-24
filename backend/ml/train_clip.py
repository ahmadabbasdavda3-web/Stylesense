"""
Fine-tune CLIP ViT-B/32 for wardrobe multi-task classification.

Strategy for 61-image few-shot (Colab GPU):
  default --freeze heads-only  -> frozen CLIP vision encoder + 4 linear heads.
  This trains ~30K params, won't overfit, runs on CPU too.
  Once you have 300+ verified images, use --freeze last-2-blocks.

  Loss = w_cat*CE(cat) + w_col*CE(color) + w_pat*CE(pattern) + w_sty*CE(style)
  with inverse-frequency class weights + label smoothing.

Usage (Colab, GPU):
  !pip install torch transformers torchvision scikit-learn pillow
  !python ml/train_clip.py --csv ml/labels_trusted.csv --epochs 40 --batch-size 16 --freeze heads-only
  # with more data:
  !python ml/train_clip.py --csv ml/labels_trusted.csv --epochs 25 --freeze last-2-blocks --lr-backbone 1e-5

Output: finetuned/clip_wardrobe.pt  (auto-loaded by ai_detector.py)
"""
import argparse
import csv
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WardrobeDataset, load_rows, make_label_maps, stratified_split, class_weights


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WardrobeCLIP(nn.Module):
    """Frozen/fine-tuned CLIP vision tower + 4 classification heads."""

    def __init__(self, model_name, n_cat, n_col, n_pat, n_sty, dropout=0.3):
        super().__init__()
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained(model_name)
        dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        self.head_cat = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_cat))
        self.head_col = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_col))
        self.head_pat = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_pat))
        self.head_sty = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_sty))

    def image_embeds(self, pixel_values):
        out = self.clip.get_image_features(pixel_values=pixel_values)
        # transformers>=5 returns BaseModelOutputWithPooling, older returns tensor
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            out = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            out = out.last_hidden_state[:, 0, :]
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return out / out.norm(dim=-1, keepdim=True)

    def forward(self, pixel_values):
        e = self.image_embeds(pixel_values)
        return {
            "category": self.head_cat(e),
            "color": self.head_col(e),
            "pattern": self.head_pat(e),
            "style": self.head_sty(e),
        }


def apply_freeze(model, mode):
    vision = model.clip.vision_model
    for p in model.clip.parameters():
        p.requires_grad = False
    for head in (model.head_cat, model.head_col, model.head_pat, model.head_sty):
        for p in head.parameters():
            p.requires_grad = True
    if mode in ("last-block", "last-2-blocks", "full"):
        if mode == "full":
            for p in vision.parameters():
                p.requires_grad = True
        else:
            n = 1 if mode == "last-block" else 2
            layers = vision.encoder.layers[-n:]
            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in vision.post_layernorm.parameters():
                p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] freeze={mode} trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")


def evaluate(model, loader, device):
    model.eval()
    correct = {k: 0 for k in ("category", "color", "pattern", "style")}
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in loader:
            x = b["pixel_values"].to(device)
            out = model(x)
            loss = sum(ce(out[k], b[k].to(device)) for k in correct) / 4.0
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            for k in correct:
                correct[k] += (out[k].argmax(-1).cpu() == b[k]).sum().item()
    acc = {k: correct[k] / max(total, 1) for k in correct}
    return loss_sum / max(total, 1), acc, sum(acc.values()) / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE_DIR, "ml", "labels.csv"))
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--freeze", default="heads-only",
                    choices=["heads-only", "last-block", "last-2-blocks", "full"])
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "finetuned", "clip_wardrobe.pt"))
    ap.add_argument("--aux-csv", default="",
                    help="High-conf pseudo-label CSV (self-training). Validated NEVER on these.")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="Sampling weight of aux rows vs 1.0 for trusted rows.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    from ai_detector import CATEGORIES, COLORS, PATTERNS, STYLES
    label_maps = make_label_maps(CATEGORIES, COLORS, PATTERNS, STYLES)
    rows = load_rows(args.csv)
    print(f"[INFO] loaded {len(rows)} trusted images from {args.csv}")
    if len(rows) < 20:
        print("[ERROR] Need >=20 labeled images. Run ml/export_dataset.py first and correct labels in UI.")
        sys.exit(1)
    if args.freeze != "heads-only" and len(rows) < 200:
        print("[WARNING] Unfreezing backbone with <200 images will likely overfit.")
        print("  Consider staying with --freeze heads-only until you have more data.")

    # Validation split comes from TRUSTED rows only — pseudo-labels must
    # never validate (they would grade the model on its own homework).
    train_rows, val_rows = stratified_split(rows, args.val_split, args.seed)
    sample_weights = [1.0] * len(train_rows)
    if args.aux_csv:
        aux = load_rows(args.aux_csv)
        trusted_ids = {r.get("id") for r in rows}
        aux = [r for r in aux if r.get("id") not in trusted_ids]
        print(f"[INFO] loaded {len(aux)} aux pseudo images from {args.aux_csv} (weight={args.aux_weight})")
        train_rows = train_rows + aux
        sample_weights = sample_weights + [args.aux_weight] * len(aux)
    print(f"[INFO] train={len(train_rows)} val={len(val_rows)}")
    train_ds = WardrobeDataset(train_rows, label_maps, args.image_size, train=True)
    val_ds = WardrobeDataset(val_rows, label_maps, args.image_size, train=False)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = WardrobeCLIP(args.model, len(CATEGORIES), len(COLORS), len(PATTERNS), len(STYLES)).to(device)
    apply_freeze(model, args.freeze)

    w = {k: class_weights(train_rows, label_maps, k).to(device) for k in label_maps}
    ce = {k: nn.CrossEntropyLoss(weight=w[k], label_smoothing=0.05) for k in label_maps}
    # Task weights: category/style matter most for recommendations; color/pattern secondary
    task_w = {"category": 1.2, "style": 1.2, "color": 1.0, "pattern": 0.8}

    head_params = list(model.head_cat.parameters()) + list(model.head_col.parameters()) \
        + list(model.head_pat.parameters()) + list(model.head_sty.parameters())
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "head_" not in n]
    opt_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        opt_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_avg, bad_epochs = -1, 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for b in train_loader:
            x = b["pixel_values"].to(device)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = sum(task_w[k] * ce[k](out[k], b[k].to(device)) for k in task_w) / sum(task_w.values())
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x)
                loss = sum(task_w[k] * ce[k](out[k], b[k].to(device)) for k in task_w) / sum(task_w.values())
                loss.backward()
                opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()
        train_loss = run_loss / len(train_ds)
        val_loss, val_acc, val_avg = evaluate(model, val_loader, device)
        acc_str = " ".join(f"{k}={v:.2f}" for k, v in val_acc.items())
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.3f} "
              f"val_loss={val_loss:.3f} {acc_str} avg={val_avg:.3f}", flush=True)
        if val_avg > best_avg + 1e-4:
            best_avg, bad_epochs = val_avg, 0
            torch.save({
                "state_dict": model.state_dict(),
                "label_maps": label_maps,
                "categories": CATEGORIES, "colors": COLORS,
                "patterns": PATTERNS, "styles": STYLES,
                "model_name": args.model, "image_size": args.image_size,
                "val_acc": val_acc, "val_avg": val_avg, "epoch": epoch,
            }, args.out)
            print(f"  -> saved best ({val_avg:.3f}) to {args.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[STOP] early stopping at epoch {epoch} (best avg={best_avg:.3f})")
                break
    print(f"[DONE] best val avg={best_avg:.3f} -> {args.out}")
    print("Next: python ml/evaluate.py --csv <test.csv> --ckpt finetuned/clip_wardrobe.pt")


if __name__ == "__main__":
    main()
