"""
Shared PyTorch dataset for multi-task wardrobe fine-tuning.
One image -> 4 labels (category, color, pattern, style).

Two transform pipelines:
  train: heavy augmentation to turn 61 images into ~600 effective samples
  val/test: deterministic CLIP preprocessing only
"""
import csv
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size=224, train=True):
    # CLIP mean/std (ViT-B/32)
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class WardrobeDataset(Dataset):
    def __init__(self, rows, label_maps, image_size=224, train=True):
        """
        rows: list of dicts with keys abs_path, category, color, pattern, style
        label_maps: dict attr -> {name: idx}
        """
        self.rows = rows
        self.maps = label_maps
        self.tf = build_transforms(image_size, train=train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["abs_path"]).convert("RGB")
        x = self.tf(img)
        return {
            "pixel_values": x,
            "category": torch.tensor(self.maps["category"][r["category"]], dtype=torch.long),
            "color": torch.tensor(self.maps["color"][r["color"]], dtype=torch.long),
            "pattern": torch.tensor(self.maps["pattern"][r["pattern"]], dtype=torch.long),
            "style": torch.tensor(self.maps["style"][r["style"]], dtype=torch.long),
            "abs_path": r["abs_path"],
        }


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if os.path.exists(r["abs_path"]):
                rows.append(r)
    return rows


def make_label_maps(categories, colors, patterns, styles):
    return {
        "category": {c: i for i, c in enumerate(categories)},
        "color": {c: i for i, c in enumerate(colors)},
        "pattern": {c: i for i, c in enumerate(patterns)},
        "style": {c: i for i, c in enumerate(styles)},
    }


def stratified_split(rows, val_ratio=0.2, seed=42, stratify_key="category"):
    """Stratified split on category; falls back to random when a class has <2 samples."""
    import random
    from collections import Counter, defaultdict
    random.seed(seed)
    counts = Counter(r[stratify_key] for r in rows)
    if min(counts.values()) < 2 or len(rows) < 10:
        idx = list(range(len(rows)))
        random.shuffle(idx)
        n_val = max(1, int(len(rows) * val_ratio))
        val_idx = set(idx[:n_val])
        val = [r for i, r in enumerate(rows) if i in val_idx]
        train = [r for i, r in enumerate(rows) if i not in val_idx]
        return train, val
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        y = [r[stratify_key] for r in rows]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        tr_idx, va_idx = next(sss.split(range(len(rows)), y))
        return [rows[i] for i in tr_idx], [rows[i] for i in va_idx]
    except Exception:
        return stratified_split_fallback(rows, val_ratio, seed)


def stratified_split_fallback(rows, val_ratio, seed):
    import random
    from collections import defaultdict
    random.seed(seed)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["category"]].append(r)
    train, val = [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def class_weights(rows, label_maps, attr):
    """Inverse-frequency weights for CrossEntropy to handle shirt(21) vs t-shirt(3)."""
    from collections import Counter
    import torch
    counts = Counter(r[attr] for r in rows)
    n_classes = len(label_maps[attr])
    total = len(rows)
    w = torch.ones(n_classes)
    for name, idx in label_maps[attr].items():
        c = counts.get(name, 0)
        w[idx] = total / (n_classes * max(c, 1))
    return w
