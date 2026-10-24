from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path.cwd()
ALIGN = ROOT / "eukaryote_model_datasets" / "features" / "unified_v3_aligned_features"
MANIFEST = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human" / "joint" / "all_15_species"
OUT_ROOT = ROOT / "eukaryote_model_results" / "unified_transformer_fusion"

# Label-derived neighbor propagation is intentionally excluded from this first run.
GROUPS = [
    "esm2", "prott5", "basic_seq", "go_ipr_pfam", "kmer_graph",
    "homology30", "homology50", "string_network",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_group(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = ALIGN / name
    fname = "train_emb_aligned.npy"
    x = np.asarray(np.load(d / fname, mmap_mode="r"), dtype=np.float32)
    present = np.load(d / "train_present.npy", mmap_mode="r").astype(np.float32)
    return x, present


def load_split(split: str, master_pos: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(MANIFEST / f"{split}_metadata.tsv", sep="\t")
    idx = np.asarray([master_pos[str(x)] for x in df["record_id"]], dtype=np.int64)
    y = (df["label"].astype(str) == "essential").astype(np.float32).to_numpy()
    return idx, y


class FusionModel(nn.Module):
    def __init__(self, dims: list[int], d_model: int = 256, nhead: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim + 1, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            for dim in dims
        ])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality = nn.Parameter(torch.zeros(1, len(dims) + 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
        nn.init.normal_(self.modality, std=0.02)

    def forward(self, xs: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        tokens = [proj(torch.cat([x, m[:, None]], dim=1)) for proj, x, m in zip(self.projections, xs, masks)]
        cls = self.cls.expand(xs[0].shape[0], -1, -1)
        z = torch.cat([cls, torch.stack(tokens, dim=1)], dim=1) + self.modality
        return self.head(self.encoder(z)[:, 0]).squeeze(1)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold), "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)), "ACC": float(accuracy_score(y, pred)),
        "Balanced_ACC": float(balanced_accuracy_score(y, pred)), "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)), "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    scores = [matthews_corrcoef(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    seed_all(args.seed)
    out = OUT_ROOT / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_ids = np.asarray(np.load(ALIGN / "master_train_ids.npy", allow_pickle=True)).astype(str)
    master_pos = {rid: i for i, rid in enumerate(master_ids)}
    split_data = {s: load_split(s, master_pos) for s in ("train", "val", "test")}
    split_idx = {s: split_data[s][0] for s in split_data}
    split_y = {s: split_data[s][1] for s in split_data}

    raw = {g: load_group(g) for g in GROUPS}
    train_idx = split_idx["train"]
    stats = {}
    arrays = []
    present = []
    dims = []
    for g in GROUPS:
        x, m = raw[g]
        # Fit imputation and scaling on training rows only; missingness is retained as a separate input.
        train_x = np.asarray(x[train_idx], dtype=np.float32)
        train_m = m[train_idx]
        valid = train_m[:, None] > 0
        sums = np.where(valid, train_x, 0.0).sum(axis=0)
        counts = valid.sum(axis=0).clip(min=1)
        mean = sums / counts
        centered = np.where(valid, train_x - mean[None, :], 0.0)
        std = np.sqrt((centered * centered).sum(axis=0) / counts).clip(min=1e-6)
        stats[g] = {"mean": mean.tolist(), "std": std.tolist()}
        z = np.where(m[:, None] > 0, x, mean[None, :])
        z = ((z - mean[None, :]) / std[None, :]).astype(np.float32)
        arrays.append(z)
        present.append(m.astype(np.float32))
        dims.append(z.shape[1])

    def tensors(split: str):
        idx = split_idx[split]
        xs = [torch.from_numpy(np.asarray(a[idx], dtype=np.float32).copy()) for a in arrays]
        ms = [torch.from_numpy(np.asarray(m[idx], dtype=np.float32).copy()) for m in present]
        y = torch.from_numpy(np.asarray(split_y[split], dtype=np.float32).copy())
        return xs, ms, y

    tx, tm, ty = tensors("train")
    vx, vm, vy = tensors("val")
    testx, testm, testy = tensors("test")
    train_ds = TensorDataset(*tx, *tm, ty)
    val_ds = TensorDataset(*vx, *vm, vy)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    model = FusionModel(dims).to(device)
    pos_weight = torch.tensor([(len(ty) - float(ty.sum())) / max(float(ty.sum()), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -np.inf
    wait = 0
    history = []

    def predict(loader: DataLoader) -> np.ndarray:
        model.eval(); out = []
        with torch.no_grad():
            for batch in loader:
                xs = [v.to(device, non_blocking=True) for v in batch[:len(GROUPS)]]
                ms = [v.to(device, non_blocking=True) for v in batch[len(GROUPS):2 * len(GROUPS)]]
                out.append(torch.sigmoid(model(xs, ms)).cpu().numpy())
        return np.concatenate(out)

    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            xs = [v.to(device, non_blocking=True) for v in batch[:len(GROUPS)]]
            ms = [v.to(device, non_blocking=True) for v in batch[len(GROUPS):2 * len(GROUPS)]]
            y = batch[-1].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(xs, ms), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        vp = predict(val_loader)
        va = roc_auc_score(vy.numpy(), vp)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_AUROC": float(va)})
        if va > best:
            best = va; wait = 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_AUROC={va:.5f}", flush=True)
        if wait >= args.patience:
            break

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device, weights_only=True))
    vp = predict(val_loader)
    test_loader = DataLoader(TensorDataset(*testx, *testm, testy), batch_size=args.batch_size * 2, shuffle=False)
    tp = predict(test_loader)
    threshold = best_threshold(vy.numpy(), vp)
    result = {"device": str(device), "groups": GROUPS, "dims": dims, "best_epoch": int(np.argmax([h["val_AUROC"] for h in history]) + 1),
              "val": metrics(vy.numpy(), vp, threshold), "test": metrics(testy.numpy(), tp, threshold)}
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "normalization_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "val_prob.npy", vp); np.save(out / "test_prob.npy", tp)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

