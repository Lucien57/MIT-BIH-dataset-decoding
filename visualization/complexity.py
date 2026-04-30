#!/usr/bin/env python3
"""Compute parameter count and MAC complexity for the ECG models.

Expected project layout:
    project_root/
        data/dataset_raw.npy
        configs/LightCNN.yaml
        configs/LSTM.yaml
        configs/NM2019.yaml
        configs/Transformer.yaml
        configs/TransformerCNN.yaml
        models/...
        util/load_model.py
        complexity.py

Run from the project root:
    python complexity.py

The reported MACs are for one input sample from this dataset. The script infers
input shape from data/dataset_raw.npy and profiles the full forward pass.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml


MODEL_SPECS = [
    ("LightCNN", "LightCNN.yaml"),
    ("LSTM", "LSTM.yaml"),
    ("NM2019", "NM2019.yaml"),
    ("Transformer", "Transformer.yaml"),
    ("TransformerCNN", "TransformerCNN.yaml"),
]


def find_root() -> Path:
    """Allow the script to be placed either at project root or in a subfolder."""
    root = Path(__file__).resolve().parent
    if (root / "configs").exists() and (root / "data").exists():
        return root
    if (root.parent / "configs").exists() and (root.parent / "data").exists():
        return root.parent
    return root


ROOT_DIR = find_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from util.load_model import load_model  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    from utils.load_model import load_model  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Params and MMACs for all ECG models.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset_raw.npy.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing model YAML configs.")
    parser.add_argument("--output", type=str, default="figures/complexity.csv", help="CSV file for the results.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for the dummy forward pass, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset: LightCNN LSTM NM2019 Transformer TransformerCNN.")
    parser.add_argument("--strict", action="store_true", help="Raise an error instead of skipping failed models.")
    parser.add_argument("--include-bn", action="store_true", help="Also count BatchNorm multiply-adds. Default: off.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_input_shape(data_dir: Path) -> Tuple[int, ...]:
    x_path = data_dir / "dataset_raw.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing data file: {x_path}")

    x = np.load(x_path, mmap_mode="r")
    if len(x.shape) < 2:
        raise ValueError(f"Expected dataset_raw.npy to have shape (N, ...), but got {x.shape}")
    return tuple(int(v) for v in x.shape[1:])


def first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def tensor_batch_and_len(x: torch.Tensor, batch_first: bool) -> Tuple[int, int]:
    if x.dim() == 2:  # unbatched: (L, E)
        return 1, int(x.shape[0])
    if batch_first:
        return int(x.shape[0]), int(x.shape[1])
    return int(x.shape[1]), int(x.shape[0])


def add_hooks(model: nn.Module, include_bn: bool = False):
    handles = []

    def add(module: nn.Module, value: int) -> None:
        module.__macs__ = getattr(module, "__macs__", 0) + int(value)

    def conv_hook(module: nn.Module, inputs, outputs) -> None:
        out = first_tensor(outputs)
        if out is None:
            return
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
            add(module, int(out.numel()) * kernel_ops)
        elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            kernel_ops = int(np.prod(module.kernel_size)) * (module.out_channels // module.groups)
            add(module, int(out.numel()) * kernel_ops)

    def linear_hook(module: nn.Linear, inputs, outputs) -> None:
        out = first_tensor(outputs)
        if out is None:
            return
        add(module, int(out.numel()) * int(module.in_features))

    def bn_hook(module: nn.Module, inputs, outputs) -> None:
        out = first_tensor(outputs)
        if out is None:
            return
        # Approximate one multiply and one add per element.
        add(module, int(out.numel()) * 2)

    def lstm_hook(module: nn.LSTM, inputs, outputs) -> None:
        x = first_tensor(inputs)
        if x is None:
            return
        batch_first = bool(module.batch_first)
        batch_size, seq_len = tensor_batch_and_len(x, batch_first)
        num_directions = 2 if module.bidirectional else 1
        hidden_size = int(module.hidden_size)
        input_size = int(module.input_size)
        total_macs = 0
        for layer in range(module.num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * num_directions
            per_direction = 4 * hidden_size * (layer_input_size + hidden_size)
            total_macs += batch_size * seq_len * num_directions * per_direction
        add(module, total_macs)

    def gru_hook(module: nn.GRU, inputs, outputs) -> None:
        x = first_tensor(inputs)
        if x is None:
            return
        batch_first = bool(module.batch_first)
        batch_size, seq_len = tensor_batch_and_len(x, batch_first)
        num_directions = 2 if module.bidirectional else 1
        hidden_size = int(module.hidden_size)
        input_size = int(module.input_size)
        total_macs = 0
        for layer in range(module.num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * num_directions
            per_direction = 3 * hidden_size * (layer_input_size + hidden_size)
            total_macs += batch_size * seq_len * num_directions * per_direction
        add(module, total_macs)

    def mha_hook(module: nn.MultiheadAttention, inputs, outputs) -> None:
        if len(inputs) < 3:
            return
        query, key, value = inputs[0], inputs[1], inputs[2]
        if not (torch.is_tensor(query) and torch.is_tensor(key) and torch.is_tensor(value)):
            return

        batch_first = bool(getattr(module, "batch_first", False))
        batch_size, target_len = tensor_batch_and_len(query, batch_first)
        _, source_len = tensor_batch_and_len(key, batch_first)

        embed_dim = int(module.embed_dim)
        num_heads = int(module.num_heads)
        head_dim = embed_dim // num_heads
        kdim = int(module.kdim) if module.kdim is not None else embed_dim
        vdim = int(module.vdim) if module.vdim is not None else embed_dim

        # Q/K/V projections + attention score matmul + weighted sum + output projection.
        q_proj = batch_size * target_len * embed_dim * embed_dim
        k_proj = batch_size * source_len * kdim * embed_dim
        v_proj = batch_size * source_len * vdim * embed_dim
        attn_scores = batch_size * num_heads * target_len * source_len * head_dim
        attn_values = batch_size * num_heads * target_len * source_len * head_dim
        out_proj = batch_size * target_len * embed_dim * embed_dim
        add(module, q_proj + k_proj + v_proj + attn_scores + attn_values + out_proj)

    for module in model.modules():
        module.__macs__ = 0
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.LSTM):
            handles.append(module.register_forward_hook(lstm_hook))
        elif isinstance(module, nn.GRU):
            handles.append(module.register_forward_hook(gru_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(mha_hook))
        elif include_bn and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            handles.append(module.register_forward_hook(bn_hook))

    return handles


def profile_macs(model: nn.Module, dummy: torch.Tensor, include_bn: bool = False) -> int:
    model.eval()
    handles = add_hooks(model, include_bn=include_bn)
    try:
        with torch.no_grad():
            _ = model(dummy)
    finally:
        for handle in handles:
            handle.remove()

    total_macs = sum(int(getattr(module, "__macs__", 0)) for module in model.modules())
    for module in model.modules():
        if hasattr(module, "__macs__"):
            delattr(module, "__macs__")
    return int(total_macs)


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def print_table(rows: List[Dict]) -> None:
    headers = ["Model", "Input", "Trainable Params", "Params(M)", "MACs", "MMACs"]
    table = []
    for r in rows:
        table.append([
            r["model"],
            r["input_shape"],
            format_int(r.get("trainable_params")),
            format_float(r.get("params_m")),
            format_int(r.get("macs")),
            format_float(r.get("mmacs")),
        ])

    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(str(x))) for w, x in zip(widths, row)]

    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for row in table:
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))


def save_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "input_shape",
        "trainable_params",
        "total_params",
        "params_m",
        "macs",
        "mmacs",
        "status",
        "error",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def run_one_model(model_name: str, cfg_path: Path, dummy: torch.Tensor, include_bn: bool) -> Dict:
    cfg = read_yaml(cfg_path)
    model = load_model(cfg["model"]).to(dummy.device)

    macs = profile_macs(model, dummy, include_bn=include_bn)
    trainable_params, total_params = count_parameters(model)

    return {
        "model": model_name,
        "input_shape": str(tuple(dummy.shape)),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "params_m": trainable_params / 1e6,
        "macs": macs,
        "mmacs": macs / 1e6,
        "status": "ok",
        "error": "",
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = ROOT_DIR / args.data_dir
    config_dir = ROOT_DIR / args.config_dir
    output_path = ROOT_DIR / args.output

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    input_shape = infer_input_shape(data_dir)
    dummy = torch.randn((1,) + input_shape, dtype=torch.float32, device=device)

    selected = set(args.models) if args.models else None
    rows: List[Dict] = []

    print(f"[INFO] Project root: {ROOT_DIR}")
    print(f"[INFO] Input shape inferred from dataset_raw.npy: {tuple(dummy.shape)}")
    print(f"[INFO] Device: {device}")

    for model_name, cfg_name in MODEL_SPECS:
        if selected is not None and model_name not in selected:
            continue

        cfg_path = config_dir / cfg_name
        if not cfg_path.exists():
            msg = f"Config not found: {cfg_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(f"[WARN] {model_name}: {msg}")
            rows.append({
                "model": model_name,
                "input_shape": str(tuple(dummy.shape)),
                "trainable_params": None,
                "total_params": None,
                "params_m": None,
                "macs": None,
                "mmacs": None,
                "status": "skipped",
                "error": msg,
            })
            continue

        print(f"[INFO] Profiling {model_name}...")
        try:
            rows.append(run_one_model(model_name, cfg_path, dummy, args.include_bn))
        except Exception as exc:
            if args.strict:
                raise
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] {model_name}: failed: {msg}")
            rows.append({
                "model": model_name,
                "input_shape": str(tuple(dummy.shape)),
                "trainable_params": None,
                "total_params": None,
                "params_m": None,
                "macs": None,
                "mmacs": None,
                "status": "failed",
                "error": msg,
            })

    print()
    print_table(rows)
    save_csv(rows, output_path)
    print(f"\n[DONE] Saved results to: {output_path}")


if __name__ == "__main__":
    main()
