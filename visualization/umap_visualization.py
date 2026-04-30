#!/usr/bin/env python3
"""UMAP visualizations for the MIT-BIH ECG project.

This script generates:
1. UMAP of the raw ECG heartbeat windows from data/dataset_raw.npy.
2. One UMAP figure per model, using the feature tensor immediately before the final classifier/head.

Expected project layout:
    project_root/
        data/dataset_raw.npy
        data/labelset_raw.npy
        configs/LightCNN.yaml
        configs/LSTM.yaml
        configs/NM2019.yaml
        configs/Transformer.yaml
        configs/TransformerCNN.yaml
        best_model/LightCNN.pt
        best_model/LSTM.pt
        best_model/NM2019.pt
        best_model/Transformer.pt
        best_model/TransformerCNN.pt
        UMAP/umap_visualization.py

Run from the project root:
    python UMAP/umap_visualization.py --device cuda:0
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The package 'umap-learn' is required. Install it with:\n"
        "    pip install umap-learn\n"
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from util.load_model import load_model  # noqa: E402


CLASS_NAMES = ["N", "A", "V", "L", "R"]

# Display name, config path, possible weight filenames under --weight-dir.
MODEL_SPECS = [
    ("LightCNN", "LightCNN.yaml", ["LightCNN.pt"]),
    ("LSTM", "LSTM.yaml", ["LSTM.pt"]),
    ("NM2019", "NM2019.yaml", ["NM2019.pt"]),
    ("Transformer", "Transformer.yaml", ["Transformer.pt"]),
    ("TransformerCNN", "TransformerCNN.yaml", ["TransformerCNN.pt"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UMAP figures for raw ECG and model features.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset_raw.npy and labelset_raw.npy.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing model YAML configs.")
    parser.add_argument("--weight-dir", type=str, default="best_model", help="Directory containing trained model weights.")
    parser.add_argument("--fig-dir", type=str, default="figures", help="Directory where UMAP figures will be saved.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for feature extraction, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling and UMAP.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Maximum samples used for UMAP. Use <=0 to use all samples, but this may be slow.",
    )
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.10, help="UMAP min_dist.")
    parser.add_argument("--metric", type=str, default="euclidean", help="UMAP metric.")
    parser.add_argument(
        "--raw-normalize",
        type=str,
        default="zscore_sample",
        choices=["none", "zscore_sample", "minmax_sample"],
        help="Normalization used before raw-data UMAP.",
    )
    parser.add_argument(
        "--no-standardize-umap-input",
        action="store_true",
        help="Disable StandardScaler before UMAP. By default, raw vectors/features are standardized before UMAP.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models to visualize. Choices: LightCNN LSTM NM2019 Transformer TransformerCNN.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a config or weight file is missing. Default: skip missing models with a warning.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / "dataset_raw.npy"
    y_path = data_dir / "labelset_raw.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing data file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")

    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    if len(x) != len(y):
        raise ValueError(f"X and y have different lengths: {len(x)} vs {len(y)}")
    return x, y


def preprocess_x(x: np.ndarray, mode: str = "zscore_sample", clip_value: Optional[float] = None) -> np.ndarray:
    x = x.astype(np.float32, copy=True)

    if mode in [None, "none"]:
        pass
    elif mode == "zscore_sample":
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x = (x - mean) / std
    elif mode == "minmax_sample":
        xmin = x.min(axis=-1, keepdims=True)
        xmax = x.max(axis=-1, keepdims=True)
        scale = np.where((xmax - xmin) < 1e-8, 1.0, xmax - xmin)
        x = (x - xmin) / scale
    else:
        raise ValueError(f"Unsupported normalization mode for this visualization script: {mode}")

    if clip_value is not None:
        x = np.clip(x, -float(clip_value), float(clip_value))
    return x


def stratified_subsample_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return idx
    sampled_idx, _ = train_test_split(
        idx,
        train_size=int(max_samples),
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return np.sort(sampled_idx)


def fit_umap(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)

    if not args.no_standardize_umap_input:
        features = StandardScaler().fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=args.metric,
        random_state=int(args.seed),
        n_components=2,
        n_jobs=1,
    )
    return reducer.fit_transform(features)


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(cls)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            alpha=0.65,
            linewidths=0,
            label=f"{name} ({int(mask.sum())})",
        )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(markerscale=3, fontsize=8, frameon=True)
    ax.grid(alpha=0.20, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined(embeddings: Dict[str, np.ndarray], labels: np.ndarray, out_path: Path) -> None:
    if not embeddings:
        return
    names = list(embeddings.keys())
    n = len(names)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        emb = embeddings[name]
        for cls in sorted(np.unique(labels)):
            mask = labels == cls
            cls_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(cls)
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=4,
                alpha=0.60,
                linewidths=0,
                label=f"{cls_name} ({int(mask.sum())})",
            )
        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(markerscale=3, fontsize=8, frameon=True)
        ax.grid(alpha=0.20, linewidth=0.5)

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_weight(weight_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for name in candidates:
        path = weight_dir / name
        if path.exists():
            return path
    return None


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            # Sometimes a raw state_dict is itself a dict of tensors.
            state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    if all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def get_features(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if not hasattr(model, "forward_features"):
        raise AttributeError(f"{model.__class__.__name__} does not define forward_features().")

    model.eval()
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).float().to(device)
            feat = model.forward_features(batch)
            feat = feat.detach().cpu().numpy()
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def run_one_model(
    display_name: str,
    config_path: Path,
    weight_path: Path,
    x_subset_original: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fig_dir: Path,
    embedding_dir: Path,
) -> Optional[np.ndarray]:
    print(f"[INFO] {display_name}: loading config {config_path}")
    cfg = read_yaml(config_path)
    data_cfg = cfg.get("data", {})
    norm_mode = data_cfg.get("normalize", "zscore_sample")
    clip_value = data_cfg.get("clip_value", None)
    x_model = preprocess_x(x_subset_original, norm_mode, clip_value)

    print(f"[INFO] {display_name}: loading model weights {weight_path}")
    model = load_model(cfg["model"]).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state = extract_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)

    print(f"[INFO] {display_name}: extracting features before final classifier")
    features = get_features(model, x_model, device, args.batch_size)
    np.save(embedding_dir / f"features_{display_name}.npy", features)

    print(f"[INFO] {display_name}: fitting UMAP on features with shape {features.shape}")
    emb = fit_umap(features, args)
    np.save(embedding_dir / f"umap_{display_name}.npy", emb)
    plot_embedding(emb, labels, f"UMAP of {display_name} features", fig_dir / f"umap_{display_name}.png")
    return emb


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = ROOT_DIR / args.data_dir
    config_dir = ROOT_DIR / args.config_dir
    weight_dir = ROOT_DIR / args.weight_dir
    fig_dir = ROOT_DIR / args.fig_dir
    embedding_dir = fig_dir / "umap_embeddings"
    fig_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] Using device: {device}")

    x, y = load_arrays(data_dir)
    subset_idx = stratified_subsample_indices(y, args.max_samples, args.seed)
    x_subset = x[subset_idx]
    y_subset = y[subset_idx]

    print(f"[INFO] Loaded data: X={x.shape}, y={y.shape}")
    print(f"[INFO] UMAP subset: {len(subset_idx)} samples")
    print(f"[INFO] Class counts in subset: {np.bincount(y_subset, minlength=len(CLASS_NAMES)).tolist()}")

    np.save(embedding_dir / "subset_indices.npy", subset_idx)
    with open(embedding_dir / "umap_settings.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1. Raw-data UMAP.
    print("[INFO] Raw ECG: fitting UMAP")
    x_raw_for_umap = preprocess_x(x_subset, args.raw_normalize, clip_value=None).reshape(len(x_subset), -1)
    raw_emb = fit_umap(x_raw_for_umap, args)
    np.save(embedding_dir / "umap_raw_data.npy", raw_emb)
    plot_embedding(raw_emb, y_subset, "UMAP of raw ECG heartbeat windows", fig_dir / "umap_raw_data.png")

    # 2. Model feature UMAPs.
    selected = set(args.models) if args.models else None
    for display_name, cfg_name, weight_candidates in MODEL_SPECS:
        if selected is not None and display_name not in selected:
            continue

        cfg_path = config_dir / cfg_name
        weight_path = find_weight(weight_dir, weight_candidates)
        if not cfg_path.exists():
            msg = f"[WARN] {display_name}: config not found: {cfg_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue
        if weight_path is None:
            msg = f"[WARN] {display_name}: no weight found under {weight_dir}; tried {weight_candidates}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        run_one_model(
            display_name=display_name,
            config_path=cfg_path,
            weight_path=weight_path,
            x_subset_original=x_subset,
            labels=y_subset,
            args=args,
            device=device,
            fig_dir=fig_dir,
            embedding_dir=embedding_dir,
        )

    print(f"[DONE] Figures saved to: {fig_dir}")
    print(f"[DONE] Embeddings/features saved to: {embedding_dir}")


if __name__ == "__main__":
    main()
