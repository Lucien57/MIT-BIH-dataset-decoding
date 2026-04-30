import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


def fmt_params(p):
    if p >= 1e6:
        return f"{p / 1e6:.2f}M"
    elif p >= 1e3:
        return f"{p / 1e3:.2f}K"
    else:
        return f"{int(p)}"


def main():
    # -------------------------
    # settings
    # -------------------------
    ACC_AS_PERCENT = True
    USE_LOG_X = True
    size_scale = 100.0
    max_bubble = 3000
    min_bubble = 60

    # -------------------------
    # data: MMACs + Params + ACC(mean, %)
    # NOTE: values below are placeholders for layout/debugging.
    # Replace them with real measured results later.
    # -------------------------
    models = [
        {"name": "LightCNN",       "mmacs": 2.563,   "params": 44430,    "acc": 95.11},
        {"name": "LSTM",           "mmacs": 95.857,   "params": 324037,    "acc": 97.97},
        {"name": "NM2019",         "mmacs": 47.318,  "params": 10462597,   "acc": 98.33},
        {"name": "Transformer",    "mmacs": 13.885,   "params": 101253,   "acc": 97.55},
        {"name": "TransformerCNN", "mmacs": 8.161,   "params": 85869,   "acc": 98.02},
    ]

    names = [m["name"] for m in models]
    mmacs = np.array([m["mmacs"] for m in models], dtype=float)
    params = np.array([m["params"] for m in models], dtype=float)
    acc = np.array([m["acc"] for m in models], dtype=float)

    # bubble size: sqrt scaling, based on params in K
    params_k = params / 1e3
    sizes = size_scale * np.sqrt(params_k)
    sizes = np.clip(sizes, min_bubble, max_bubble)

    color_map = {
        "LightCNN": "#FFD869",
        "LSTM": "#B7E4FF",
        "NM2019": "#F58787",
        "Transformer": "#D1AEEC",
        "TransformerCNN": "#91DBB2",
    }

    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=500)

    # label offsets; adjust later if real values overlap
    dy_map = {
        "LightCNN": 0.20,
        "LSTM": 0.22,
        "NM2019": 0.22,
        "Transformer": 0.20,
        "TransformerCNN": 0.22,
    }
    default_dy = 0.18

    for n, x, y, s, p in zip(names, mmacs, acc, sizes, params):
        c = color_map.get(n, "#CCCCCC")
        edge = "white"
        lw = 0.8

        ax.scatter(x, y, s=s, c=c, edgecolors=edge, linewidths=lw, alpha=0.9)
        dy = dy_map.get(n, default_dy)
        ax.text(x, y + dy, n, ha="center", va="bottom", fontsize=8)
        ax.text(x, y + 0.08, f"{fmt_params(p)}", ha="center", va="top", fontsize=8)

    ax.set_xlabel("MACs (M)", fontsize=12)
    ax.set_ylabel("ACC (%)" if ACC_AS_PERCENT else "ACC", fontsize=12)

    if USE_LOG_X:
        ax.set_xscale("log")
        xticks = [1, 2, 3, 5, 7, 10, 15, 20, 30]
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        ax.minorticks_on()
        ax.tick_params(axis="x", which="major", labelsize=10)

    x_min, x_max = mmacs.min(), mmacs.max()
    if USE_LOG_X:
        ax.set_xlim(x_min * 0.70, x_max * 1.35)
    else:
        x_margin = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    y_min, y_max = acc.min(), acc.max()
    y_margin = (y_max - y_min) * 0.28 if y_max > y_min else 0.5
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(False, which="minor")

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")

    from matplotlib.lines import Line2D

    legend_handles = []
    for n in names:
        c = color_map.get(n, "#CCCCCC")
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o", linestyle="None",
                markerfacecolor=c,
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=6,
                label=n,
            )
        )

    leg = ax.legend(
        handles=legend_handles,
        fontsize=8,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_boxstyle("round", pad=0, rounding_size=1.0)

    plt.tight_layout(pad=0.2)

    os.makedirs("./figures", exist_ok=True)
    fig.savefig("./figures/macs_param_ecg.png", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    main()
