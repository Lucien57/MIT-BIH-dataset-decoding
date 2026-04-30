import argparse
import copy
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from util.load_data import build_dataloaders
from util.load_model import count_parameters, load_model


CLASS_NAMES = ["N", "A", "V", "L", "R"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECG classifier on MIT-BIH heartbeat windows.")
    parser.add_argument("--config", type=str, default="configs/NM2019.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override config train.device, e.g. cuda:1 or cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed and data.split_seed.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override config data.data_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config train.batch_size.")
    parser.add_argument("--fold", type=int, default=None, help="Run one stratified k-fold fold. Default uses config data.fold.")
    parser.add_argument("--n-splits", type=int, default=None, help="Override config data.n_splits.")
    parser.add_argument("--all-folds", action="store_true", help="Run all folds sequentially in this process.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict, args) -> Dict:
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
        cfg["data"]["split_seed"] = args.seed
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.n_splits is not None:
        cfg["data"]["n_splits"] = args.n_splits
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_dirs(cfg: Dict) -> Tuple[Path, Path]:
    exp_name = cfg["experiment"].get("name", "ECG_experiment")
    seed = int(cfg["experiment"].get("seed", 0))
    split_mode = cfg["data"].get("split_mode", "stratified_kfold").lower()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if split_mode in ["kfold", "stratified_kfold", "stratified-kfold"]:
        fold_tag = f"fold_{int(cfg['data'].get('fold', 0)):02d}"
    else:
        fold_tag = "holdout"

    save_dir = Path(cfg["experiment"].get("save_dir", "saved")) / exp_name / f"seed_{seed}" / fold_tag / timestamp
    log_dir = Path(cfg["experiment"].get("log_dir", "logs")) / exp_name / f"seed_{seed}" / fold_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, log_dir / f"{timestamp}.log"


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def make_optimizer(model: nn.Module, train_cfg: Dict):
    name = train_cfg.get("optimizer", "AdamW")
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(optimizer, sched_cfg: Dict, train_cfg: Dict):
    name = sched_cfg.get("name", "none")
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.1)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 80)),
            eta_min=float(sched_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def make_class_weights(y_train: np.ndarray, n_class: int, mode: str, device: torch.device):
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class_weight mode: {mode}")
    counts = np.bincount(y_train, minlength=n_class).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, train_cfg, logger, epoch):
    model.train()
    total_loss, total_count = 0.0, 0
    correct = 0
    print_freq = int(train_cfg.get("print_freq", 50))
    grad_clip_norm = train_cfg.get("grad_clip_norm", None)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()

        if print_freq > 0 and step % print_freq == 0:
            logger.info(f"Epoch {epoch:03d} | step {step:04d}/{len(loader):04d} | "
                        f"loss {total_loss / total_count:.4f} | acc {correct / total_count:.4f}")

    return total_loss / total_count, correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_class: int):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_logits, all_y = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_count += y.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    prob = torch.softmax(logits, dim=1).numpy()
    y_pred = prob.argmax(axis=1)

    metrics = {
        "loss": float(total_loss / total_count),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        present_classes = np.unique(y_true)
        if len(present_classes) == n_class:
            metrics["macro_auc_ovr"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
            metrics["weighted_auc_ovr"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="weighted"))
    except ValueError:
        pass

    return metrics, y_true, y_pred, prob


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def log_run_info(logger, cfg, info, model):
    logger.info("========== Run Info ==========")
    logger.info(f"Experiment: {cfg['experiment'].get('name', 'ECG_experiment')}")
    logger.info(f"Model: {cfg['model'].get('name', 'NM2019')}")
    logger.info(f"Dataset: {cfg['data'].get('x_file')} + {cfg['data'].get('y_file')}")
    logger.info(f"Split: {info['split_mode']} | n_splits={info['n_splits']} | test_fold={info['fold']} | val_fold={info['val_fold']}")
    logger.info(f"Samples: total={info['n_samples']} | train={info['n_train']} | val={info['n_val']} | test={info['n_test']}")
    logger.info(f"Classes: {info['n_class']} | Channels: {info['in_channels']} | Time points: {info['input_length']}")
    logger.info(f"Class counts total: {info['class_counts_total']}")
    logger.info(f"Class counts train: {info['class_counts_train']}")
    logger.info(f"Class counts val: {info['class_counts_val']}")
    logger.info(f"Class counts test: {info['class_counts_test']}")
    logger.info(f"Batch size: {cfg['train'].get('batch_size')} | LR: {cfg['train'].get('learning_rate')} | "
                f"Weight decay: {cfg['train'].get('weight_decay')}")
    logger.info(f"Normalization: {cfg['data'].get('normalize')} | Augmentation: {cfg['augmentation'].get('enable')}")
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    logger.info("================================")


def run_one_fold(cfg: Dict, args) -> Dict:
    seed = int(cfg["experiment"].get("seed", 0))
    set_seed(seed)

    save_dir, log_path = prepare_dirs(cfg)
    logger = setup_logger(log_path)
    shutil.copy(args.config, save_dir / "config_original.yaml")
    save_json(cfg, save_dir / "config_used.json")

    loaders, info, y_train = build_dataloaders(cfg["data"], cfg.get("augmentation", {}), cfg["train"])
    save_json({k: v for k, v in info.items() if not k.endswith("indices")}, save_dir / "split_info.json")
    save_json({"train": info["train_indices"], "val": info["val_indices"], "test": info["test_indices"]}, save_dir / "split_indices.json")

    device = torch.device(cfg["train"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    # Keep model config synchronized with data shape/classes even if config is edited.
    cfg["model"]["n_class"] = info["n_class"]
    cfg["model"]["in_channels"] = info["in_channels"]
    model = load_model(cfg["model"]).to(device)

    log_run_info(logger, cfg, info, model)

    class_weights = make_class_weights(
        y_train, info["n_class"], cfg["train"].get("class_weight", "none"), device
    )
    if class_weights is not None:
        logger.info(f"Class weights: {class_weights.detach().cpu().numpy().round(4).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = make_optimizer(model, cfg["train"])
    scheduler = make_scheduler(optimizer, cfg.get("scheduler", {"name": "none"}), cfg["train"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    epochs = int(cfg["train"].get("epochs", 80))
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 15))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, cfg["train"], logger, epoch
        )
        val_metrics, _, _, _ = evaluate(model, loaders["val"], criterion, device, info["n_class"])

        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:03d} | lr {lr:.2e} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                    f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} "
                    f"macroF1 {val_metrics['macro_f1']:.4f} weightedF1 {val_metrics['weighted_f1']:.4f}")

        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        if scheduler is not None:
            if cfg.get("scheduler", {}).get("name") == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, save_dir / "best_model.pt")
            logger.info(f"Saved best model at epoch {epoch:03d}.")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                logger.info(f"Early stopping: no val-loss improvement for {early_stop_patience} epochs.")
                break

    save_json(history, save_dir / "history.json")
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, save_dir / "last_model.pt")

    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics, y_true, y_pred, prob = evaluate(model, loaders["test"], criterion, device, info["n_class"])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(info["n_class"])))
    np.savetxt(save_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    np.save(save_dir / "test_prob.npy", prob)

    result = {
        "model": cfg["model"].get("name", "model"),
        "seed": int(cfg["experiment"].get("seed", 0)),
        "split_mode": info["split_mode"],
        "fold": int(info["fold"]),
        "val_fold": int(info["val_fold"]),
        "best_epoch": int(best_epoch),
        "save_dir": str(save_dir),
    }
    result.update(test_metrics)
    save_json(result, save_dir / "test_metrics.json")

    names = CLASS_NAMES[:info["n_class"]] if info["n_class"] <= len(CLASS_NAMES) else [str(i) for i in range(info["n_class"])]
    report = classification_report(y_true, y_pred, labels=list(range(info["n_class"])),
                                   target_names=names, digits=4, zero_division=0)
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)

    logger.info("========== Test Results ==========")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
    logger.info("Confusion matrix:\n" + np.array2string(cm))
    logger.info("Classification report:\n" + report)
    logger.info(f"Artifacts saved to: {save_dir}")
    return result


def summarize_kfold(results):
    metric_names = ["acc", "balanced_acc", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    summary = {"fold_results": results, "mean_std": {}}
    for name in metric_names:
        vals = [r[name] for r in results if name in r]
        if len(vals) > 0:
            summary["mean_std"][name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
    return summary


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    if args.all_folds:
        n_splits = int(cfg["data"].get("n_splits", 10))
        results = []
        for fold in range(n_splits):
            fold_cfg = copy.deepcopy(cfg)
            fold_cfg["data"]["fold"] = fold
            results.append(run_one_fold(fold_cfg, args))

        summary = summarize_kfold(results)
        exp_name = cfg["experiment"].get("name", "ECG_experiment")
        seed = int(cfg["experiment"].get("seed", 0))
        out_dir = Path(cfg["experiment"].get("log_dir", "logs")) / exp_name / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"kfold_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
        save_json(summary, out_path)

        print("========== K-Fold Summary ==========")
        for k, v in summary["mean_std"].items():
            print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
        print(f"Saved k-fold summary to: {out_path}")
    else:
        run_one_fold(cfg, args)


if __name__ == "__main__":
    main()
