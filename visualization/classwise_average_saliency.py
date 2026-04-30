#!/usr/bin/env python3
"""
Class-wise average temporal saliency curves for the MIT-BIH ECG project.

Place this folder under the project root, then run for example:
    python Saliency/classwise_average_saliency.py --device cuda:0

Expected project layout:
    project_root/
      data/dataset_raw.npy
      data/labelset_raw.npy
      configs/*.yaml
      models/*.py
      util/*.py
      best_model/NM2019.pt
      best_model/LSTM.pt
      best_model/LightCNN.pt
      best_model/Transformer.pt
      best_model/LightTransformerCNN.pt
      Saliency/classwise_average_saliency.py

The script computes input-gradient saliency over time:
    saliency(t) = mean_channel | d logit_target / d x(t) |

For single-lead ECG, this gives a temporal importance curve instead of an EEG-style
channel topomap.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from util.load_data import load_arrays, make_splits  # noqa: E402
from util.load_model import load_model  # noqa: E402


MODEL_CONFIGS = {
    "NM2019": "NM2019.yaml",
    "LSTM": "LSTM.yaml",
    "LightCNN": "LightCNN.yaml",
    "Transformer": "Transformer.yaml",
    "TransformerCNN": "TransformerCNN.yaml",
}

MODEL_WEIGHTS = {
    "NM2019": "NM2019.pt",
    "LSTM": "LSTM.pt",
    "LightCNN": "LightCNN.pt",
    "Transformer": "Transformer.pt",
    "TransformerCNN": "TransformerCNN.pt",
}

CANONICAL_MODEL_TAG = {
    "ECGLSTM": "LSTM",
    "ECGTransformer": "Transformer",
}

DEFAULT_MODELS = ["NM2019", "LSTM", "LightCNN", "Transformer", "TransformerCNN"]
DEFAULT_CLASS_NAMES = ["N", "A", "V", "L", "R"]


def parse_args():
    parser = argparse.ArgumentParser(description="Class-wise average saliency curves for ECG models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model tags to visualize. Default: all five implemented models.")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--weight-dir", type=str, default="best_model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="figures/saliency")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", type=str, default="test", choices=["all", "train", "val", "test"],
                        help="Which split to use for saliency averaging. Default: test.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Override data.fold in config. Use this when weights are from a specific k-fold fold.")
    parser.add_argument("--n-splits", type=int, default=None,
                        help="Override data.n_splits in config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples-per-class", type=int, default=800,
                        help="Limit samples per class for speed. Use 0 to use all available samples.")
    parser.add_argument("--target-mode", type=str, default="true", choices=["true", "pred"],
                        help="Gradient target: true label or predicted label. Default: true.")
    parser.add_argument("--correct-only", action="store_true",
                        help="Only average samples that are correctly classified by the model.")
    parser.add_argument("--smooth-window", type=int, default=9,
                        help="Moving-average window for display. Use 1 to disable smoothing.")
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--ignore-checkpoint-config", action="store_true",
                        help="Always use YAML config even if checkpoint contains a saved cfg.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config_path(model_tag: str, config_dir: Path) -> Path:
    if model_tag not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model tag '{model_tag}'. Known: {sorted(MODEL_CONFIGS)}")
    path = config_dir / MODEL_CONFIGS[model_tag]
    if not path.exists():
        raise FileNotFoundError(f"Config not found for {model_tag}: {path}")
    return path


def get_weight_path(model_tag: str, weight_dir: Path) -> Path:
    if model_tag not in MODEL_WEIGHTS:
        raise ValueError(f"Unknown model tag '{model_tag}'. Known: {sorted(MODEL_WEIGHTS)}")
    path = weight_dir / MODEL_WEIGHTS[model_tag]
    if not path.exists():
        raise FileNotFoundError(
            f"Weight not found for {model_tag}: {path}\n"
            f"Expected files like best_model/NM2019.pt, best_model/LSTM.pt, ..."
        )
    return path


def override_cfg(cfg: Dict, args) -> Dict:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["train"] = dict(cfg.get("train", {}))
    cfg["model"] = dict(cfg.get("model", {}))

    cfg["data"]["data_dir"] = str(resolve_path(args.data_dir))
    if args.fold is not None:
        cfg["data"]["fold"] = int(args.fold)
    if args.n_splits is not None:
        cfg["data"]["n_splits"] = int(args.n_splits)
    cfg["train"]["device"] = args.device
    return cfg


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        else:
            # Plain state_dict saved directly by torch.save(model.state_dict(), path)
            if all(torch.is_tensor(v) for v in checkpoint.values()):
                state = checkpoint
            else:
                raise KeyError("Could not find 'model', 'state_dict', or 'model_state_dict' in checkpoint.")
    else:
        raise TypeError("Unsupported checkpoint format. Expected a dict/state_dict saved by torch.save.")

    state = strip_prefix_if_present(state, "module.")
    state = strip_prefix_if_present(state, "model.")
    return state


def load_model_from_checkpoint(model_tag: str, args, device: torch.device):
    config_dir = resolve_path(args.config_dir)
    weight_dir = resolve_path(args.weight_dir)

    yaml_cfg = override_cfg(load_yaml(get_config_path(model_tag, config_dir)), args)
    weight_path = get_weight_path(model_tag, weight_dir)
    checkpoint = torch.load(weight_path, map_location="cpu")

    cfg = yaml_cfg
    if isinstance(checkpoint, dict) and "cfg" in checkpoint and not args.ignore_checkpoint_config:
        # Training script saves {'model': state_dict, 'cfg': cfg, ...}. This is safest.
        cfg = override_cfg(checkpoint["cfg"], args)

    state = extract_state_dict(checkpoint)
    model = load_model(cfg["model"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] {model_tag}: loaded with strict=False")
        if missing:
            print(f"       missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"       unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    model.to(device)
    model.eval()

    display_tag = CANONICAL_MODEL_TAG.get(model_tag, model_tag)
    return model, cfg, weight_path, display_tag


def normalize_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def normalize_minmax_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xmin = x.min(axis=-1, keepdims=True)
    xmax = x.max(axis=-1, keepdims=True)
    return 2.0 * (x - xmin) / (xmax - xmin + eps) - 1.0


def normalize_by_train_stats(x: np.ndarray, x_train: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    return (x - mean) / (std + eps)


def normalize_array_for_split(x: np.ndarray, train_idx: np.ndarray, method: str) -> np.ndarray:
    method = (method or "none").lower()
    if method == "none":
        return x.astype(np.float32)
    if method == "zscore_sample":
        return normalize_sample(x).astype(np.float32)
    if method == "minmax_sample":
        return normalize_minmax_sample(x).astype(np.float32)
    if method == "zscore_train":
        return normalize_by_train_stats(x, x[train_idx]).astype(np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def prepare_data(cfg: Dict, split: str):
    x, y = load_arrays(cfg["data"])
    train_idx, val_idx, test_idx, split_info = make_splits(y, cfg["data"])

    method = cfg["data"].get("normalize", "zscore_sample")
    x_norm = normalize_array_for_split(x, train_idx, method)

    if split == "train":
        use_idx = train_idx
    elif split == "val":
        use_idx = val_idx
    elif split == "test":
        use_idx = test_idx
    elif split == "all":
        use_idx = np.arange(len(y))
    else:
        raise ValueError(split)

    return x_norm[use_idx], y[use_idx], use_idx, split_info


def choose_indices_per_class(y: np.ndarray, n_class: int, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for c in range(n_class):
        idx = np.where(y == c)[0]
        if max_per_class and max_per_class > 0 and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(idx)
    if not selected:
        return np.array([], dtype=np.int64)
    selected = np.concatenate(selected).astype(np.int64)
    rng.shuffle(selected)
    return selected


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return x
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    if x.ndim == 1:
        return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")[:len(x)]
    return np.stack([moving_average(row, window) for row in x], axis=0)


def minmax_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_saliency_for_model(model, x: np.ndarray, y: np.ndarray, args, device: torch.device,
                               n_class: int) -> Dict:
    selected_idx = choose_indices_per_class(y, n_class, args.max_samples_per_class, args.seed)
    x_sel = x[selected_idx]
    y_sel = y[selected_idx]

    all_sal = []
    all_wave = []
    all_y = []
    all_pred = []
    batch_size = int(args.batch_size)

    for start in range(0, len(selected_idx), batch_size):
        xb_np = x_sel[start:start + batch_size]
        yb_np = y_sel[start:start + batch_size]

        xb = torch.from_numpy(xb_np).float().to(device)
        yb = torch.from_numpy(yb_np).long().to(device)
        xb.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        target = pred if args.target_mode == "pred" else yb
        score = logits.gather(1, target.unsqueeze(1)).sum()
        score.backward()

        grad = xb.grad.detach().abs().mean(dim=1)  # (B, T), averaged over channel(s)
        sal = grad.cpu().numpy()
        wave = xb.detach().cpu().numpy()[:, 0, :]

        # Normalize saliency per sample before class averaging. This emphasizes where the
        # model looked within each beat, not the absolute gradient scale of each sample.
        sal = sal / (sal.max(axis=1, keepdims=True) + 1e-8)

        all_sal.append(sal)
        all_wave.append(wave)
        all_y.append(yb_np)
        all_pred.append(pred.detach().cpu().numpy())

    sal = np.concatenate(all_sal, axis=0)
    wave = np.concatenate(all_wave, axis=0)
    labels = np.concatenate(all_y, axis=0)
    preds = np.concatenate(all_pred, axis=0)

    class_sal = []
    class_wave = []
    class_counts = []
    class_correct = []

    for c in range(n_class):
        mask = labels == c
        total_c = int(mask.sum())
        correct_c = int(((preds == labels) & mask).sum())
        if args.correct_only:
            mask = mask & (preds == labels)

        if mask.sum() == 0:
            class_sal.append(np.full(sal.shape[1], np.nan, dtype=np.float32))
            class_wave.append(np.full(wave.shape[1], np.nan, dtype=np.float32))
            class_counts.append(0)
            class_correct.append(correct_c)
            continue

        mean_sal = sal[mask].mean(axis=0)
        mean_sal = moving_average(mean_sal, args.smooth_window)
        mean_sal = minmax_1d(mean_sal)

        class_sal.append(mean_sal.astype(np.float32))
        class_wave.append(wave[mask].mean(axis=0).astype(np.float32))
        class_counts.append(int(mask.sum()))
        class_correct.append(correct_c)

    return {
        "saliency": np.stack(class_sal, axis=0),
        "waveform": np.stack(class_wave, axis=0),
        "counts_used": class_counts,
        "correct_total_before_filter": class_correct,
        "labels": labels,
        "preds": preds,
        "selected_local_indices": selected_idx,
    }


def save_npz(out_path: Path, result: Dict, class_names: List[str], model_tag: str):
    np.savez_compressed(
        out_path,
        saliency=result["saliency"],
        waveform=result["waveform"],
        counts_used=np.asarray(result["counts_used"], dtype=np.int64),
        correct_total_before_filter=np.asarray(result["correct_total_before_filter"], dtype=np.int64),
        class_names=np.asarray(class_names),
        model_tag=np.asarray([model_tag]),
    )


def plot_model_saliency(model_tag: str, result: Dict, class_names: List[str], output_path: Path):
    sal = result["saliency"]
    wave = result["waveform"]
    counts = result["counts_used"]
    n_class, n_times = sal.shape
    t = np.arange(n_times)

    fig, axes = plt.subplots(n_class, 1, figsize=(8, 1.8 * n_class), sharex=True)
    if n_class == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        ax.plot(t, wave[c], linewidth=1.0, label="mean ECG")
        ax.set_ylabel(class_names[c])
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(t, sal[c], linewidth=1.0, linestyle="--", label="saliency")
        ax2.fill_between(t, 0, sal[c], alpha=0.18)
        ax2.set_ylim(0, 1.05)
        ax2.set_yticks([0, 1])
        ax2.set_ylabel("sal.")

        ax.set_title(f"Class {class_names[c]} | n={counts[c]}", fontsize=9, loc="left")

    axes[-1].set_xlabel("Time point in 300-sample heartbeat window")
    fig.suptitle(f"Class-wise average ECG waveform + temporal saliency: {model_tag}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compare_models(all_results: Dict[str, Dict], class_names: List[str], output_path: Path):
    if not all_results:
        return
    any_result = next(iter(all_results.values()))
    n_class, n_times = any_result["saliency"].shape
    t = np.arange(n_times)

    fig, axes = plt.subplots(n_class, 1, figsize=(8, 1.8 * n_class), sharex=True)
    if n_class == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        for model_tag, result in all_results.items():
            ax.plot(t, result["saliency"][c], linewidth=1.1, label=model_tag)
        ax.set_ylabel(class_names[c])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_title(f"Class {class_names[c]}", fontsize=9, loc="left")

    axes[-1].set_xlabel("Time point in 300-sample heartbeat window")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Class-wise average temporal saliency comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_mean_waveform(x: np.ndarray, y: np.ndarray, class_names: List[str], output_path: Path):
    n_class = len(class_names)
    n_times = x.shape[-1]
    t = np.arange(n_times)
    fig, axes = plt.subplots(n_class, 1, figsize=(8, 1.6 * n_class), sharex=True)
    if n_class == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        mask = y == c
        if mask.sum() == 0:
            continue
        mean_wave = x[mask, 0, :].mean(axis=0)
        std_wave = x[mask, 0, :].std(axis=0)
        ax.plot(t, mean_wave, linewidth=1.0)
        ax.fill_between(t, mean_wave - std_wave, mean_wave + std_wave, alpha=0.15)
        ax.set_ylabel(class_names[c])
        ax.set_title(f"Class {class_names[c]} | n={int(mask.sum())}", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time point in 300-sample heartbeat window")
    fig.suptitle("Class-wise mean ECG waveform", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    class_names = list(args.class_names)

    # Use the first model config to prepare the shared data split. Model configs should share data settings.
    first_cfg = override_cfg(load_yaml(get_config_path(args.models[0], resolve_path(args.config_dir))), args)
    x, y, global_indices, split_info = prepare_data(first_cfg, args.split)
    n_class = int(first_cfg["data"].get("n_class", len(np.unique(y))))
    if len(class_names) < n_class:
        class_names = class_names + [str(i) for i in range(len(class_names), n_class)]
    class_names = class_names[:n_class]

    plot_class_mean_waveform(x, y, class_names, out_dir / f"classwise_mean_waveform_{args.split}.png")

    all_results = {}
    summary = {
        "split": args.split,
        "split_info": split_info,
        "n_samples_in_split": int(len(y)),
        "global_indices_used_split": "not_saved_in_json; see arrays/*.npz selected_local_indices",
        "target_mode": args.target_mode,
        "correct_only": bool(args.correct_only),
        "max_samples_per_class": int(args.max_samples_per_class),
        "smooth_window": int(args.smooth_window),
        "models": {},
    }

    for model_tag in args.models:
        print(f"[INFO] Processing model: {model_tag}")
        model, cfg, weight_path, display_tag = load_model_from_checkpoint(model_tag, args, device)
        result = compute_saliency_for_model(model, x, y, args, device, n_class)
        all_results[display_tag] = result

        fig_path = out_dir / f"classwise_average_saliency_{display_tag}_{args.split}.png"
        arr_path = out_dir / "arrays" / f"classwise_average_saliency_{display_tag}_{args.split}.npz"
        plot_model_saliency(display_tag, result, class_names, fig_path)
        save_npz(arr_path, result, class_names, display_tag)

        total_selected = int(len(result["labels"]))
        acc_selected = float((result["preds"] == result["labels"]).mean()) if total_selected > 0 else 0.0
        summary["models"][display_tag] = {
            "weight_path": str(weight_path),
            "figure": str(fig_path),
            "array": str(arr_path),
            "counts_used_per_class": result["counts_used"],
            "correct_total_before_filter_per_class": result["correct_total_before_filter"],
            "selected_samples_total": total_selected,
            "selected_accuracy": acc_selected,
        }
        print(f"[SAVE] {fig_path}")

    compare_path = out_dir / f"classwise_average_saliency_compare_models_{args.split}.png"
    plot_compare_models(all_results, class_names, compare_path)
    summary["compare_figure"] = str(compare_path)

    with open(out_dir / "saliency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] {compare_path}")
    print(f"[DONE] Figures saved under: {out_dir}")


if __name__ == "__main__":
    main()
