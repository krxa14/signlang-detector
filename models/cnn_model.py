"""PyTorch CNN classifier on 64x64 hand crops."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "cnn_best.pt"
DEVICE = torch.device("cpu")


class SignCNN(nn.Module):
    def __init__(self, num_classes: int = 106):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


_eval_tfm = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class CNNModel:
    def __init__(self, num_classes: int = 106):
        self.num_classes = num_classes
        self.net = SignCNN(num_classes).to(DEVICE)

    def train(self, train_loader, val_loader, epochs: int = 3, lr: float = 1e-3) -> dict:
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        history = []
        for ep in range(epochs):
            self.net.train()
            total, correct, running = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = self.net(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
            val_acc = self.evaluate(val_loader)
            train_acc = correct / max(1, total)
            avg_loss = running / max(1, total)
            history.append({"epoch": ep + 1, "train_loss": avg_loss, "train_acc": train_acc, "val_acc": val_acc})
            print(f"  [CNN] epoch {ep+1}/{epochs} loss={avg_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        self.save(DEFAULT_WEIGHTS)
        return {"history": history, "weights": str(DEFAULT_WEIGHTS)}

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        self.net.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = self.net(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        return correct / max(1, total)

    @torch.no_grad()
    def predict(self, crop_bgr_or_pil) -> dict:
        self.net.eval()
        if isinstance(crop_bgr_or_pil, np.ndarray):
            img = Image.fromarray(crop_bgr_or_pil[:, :, ::-1])  # BGR->RGB
        else:
            img = crop_bgr_or_pil
        x = _eval_tfm(img).unsqueeze(0).to(DEVICE)
        out = self.net(x)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
        cls = int(probs.argmax())
        return {"class_id": cls, "confidence": float(probs[cls]), "probs": probs}

    def save(self, path: Optional[Path] = None):
        path = Path(path or DEFAULT_WEIGHTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "num_classes": self.num_classes}, path)

    def load(self, path: Optional[Path] = None):
        path = Path(path or DEFAULT_WEIGHTS)
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.num_classes = ck.get("num_classes", self.num_classes)
        self.net = SignCNN(self.num_classes).to(DEVICE)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()


if __name__ == "__main__":
    m = CNNModel()
    print(m.net)
