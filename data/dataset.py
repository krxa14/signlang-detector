"""Dataset utilities: fix train/valid split, CNN DataLoaders, class mapping."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "American-sign-language-2"
DATA_YAML = DATASET_ROOT / "data.yaml"


def _read_yaml() -> dict:
    with open(DATA_YAML) as f:
        return yaml.safe_load(f)


def get_class_mapping() -> Dict[int, str]:
    return {i: name for i, name in enumerate(_read_yaml()["names"])}


def fix_valid_split(valid_fraction: float = 0.2, seed: int = 42) -> dict:
    """Move valid_fraction of labeled train images to valid/, rewrite data.yaml."""
    train_img_dir = DATASET_ROOT / "train" / "images"
    train_lbl_dir = DATASET_ROOT / "train" / "labels"
    valid_img_dir = DATASET_ROOT / "valid" / "images"
    valid_lbl_dir = DATASET_ROOT / "valid" / "labels"
    valid_lbl_dir.mkdir(parents=True, exist_ok=True)

    # labeled train images only
    label_stems = {p.stem for p in train_lbl_dir.glob("*.txt")}
    train_imgs = [p for p in train_img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem in label_stems]

    # if valid already has labels, skip
    existing_valid_labels = list(valid_lbl_dir.glob("*.txt"))
    if len(existing_valid_labels) > 50:
        moved = 0
    else:
        random.seed(seed)
        random.shuffle(train_imgs)
        n_move = max(1, int(len(train_imgs) * valid_fraction))
        to_move = train_imgs[:n_move]
        moved = 0
        for img_path in to_move:
            lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            shutil.move(str(img_path), str(valid_img_dir / img_path.name))
            shutil.move(str(lbl_path), str(valid_lbl_dir / lbl_path.name))
            moved += 1

    # rewrite data.yaml with absolute paths + val pointing to valid
    y = _read_yaml()
    y["train"] = str((DATASET_ROOT / "train" / "images").resolve())
    y["val"] = str((DATASET_ROOT / "valid" / "images").resolve())
    y["test"] = str((DATASET_ROOT / "test" / "images").resolve())
    with open(DATA_YAML, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)

    # clear stale cache
    for cache in [train_img_dir.parent / "labels.cache", valid_img_dir.parent / "labels.cache", DATASET_ROOT / "test" / "labels.cache"]:
        if cache.exists():
            cache.unlink()

    return {
        "moved": moved,
        "train_images": len(list(train_img_dir.glob("*"))),
        "valid_images": len(list(valid_img_dir.glob("*"))),
    }


class CropClassificationDataset(Dataset):
    """Crops hand region from YOLO label boxes for CNN classification."""

    def __init__(self, img_dir: Path, lbl_dir: Path, tfm: transforms.Compose):
        self.samples: list[Tuple[Path, Path]] = []
        lbl_stems = {p.stem: p for p in lbl_dir.glob("*.txt")}
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_stems.get(img.stem)
            if lbl:
                self.samples.append((img, lbl))
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        with open(lbl_path) as f:
            line = f.readline().strip().split()
        if not line:
            return self.tfm(img), 0
        cls = int(line[0])
        xc, yc, w, h = map(float, line[1:5])
        x1 = max(0, int((xc - w / 2) * W))
        y1 = max(0, int((yc - h / 2) * H))
        x2 = min(W, int((xc + w / 2) * W))
        y2 = min(H, int((yc + h / 2) * H))
        if x2 <= x1 or y2 <= y1:
            crop = img
        else:
            crop = img.crop((x1, y1, x2, y2))
        return self.tfm(crop), cls


def get_dataloaders(batch_size: int = 64, img_size: int = 64, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_ds = CropClassificationDataset(DATASET_ROOT / "train" / "images", DATASET_ROOT / "train" / "labels", train_tfm)
    val_ds = CropClassificationDataset(DATASET_ROOT / "valid" / "images", DATASET_ROOT / "valid" / "labels", val_tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    info = fix_valid_split()
    print(f"fix_valid_split: {info}")
    print(f"classes: {len(get_class_mapping())}")
    tl, vl = get_dataloaders(batch_size=16, num_workers=0)
    print(f"train batches: {len(tl)} | val batches: {len(vl)}")
