"""
Analyze autoresearch experiment results and generate wider-search visualizations.

Usage:
    uv run analyze_results.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.tsv")
OUT_DIR = Path("analysis")
CHAMPION_PROGRESS_PATH = OUT_DIR / "champion_progress.png"
SEARCH_OVERVIEW_PATH = OUT_DIR / "search_overview.png"
CHAMPION_LOSS_PATH = OUT_DIR / "champion_loss_traces.png"
LOSS_PATTERN = re.compile(r"step\s+\d+\s+\([^)]*\)\s+\|\s+loss:\s+([0-9.]+)")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise SystemExit("results.tsv is empty")

    numeric_columns = ["experiment_id", "val_bpb", "memory_gb"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    for col in ["mode", "status", "frontier_action", "description", "log_path", "tags", "parent", "parent_b"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "experiment_id" not in df.columns:
        df["experiment_id"] = range(1, len(df) + 1)

    df = df.sort_values("experiment_id").reset_index(drop=True)
    df["is_crash"] = df["status"].str.lower() == "crash"
    df["is_champion"] = df["frontier_action"].str.lower() == "champion"
    df["is_frontier"] = df["frontier_action"].str.lower() == "frontier"
    valid = (~df["is_crash"]) & df["val_bpb"].notna() & (df["val_bpb"] > 0)
    df["running_best"] = df["val_bpb"].where(valid).cummin()
    return df


def champion_rows(df: pd.DataFrame) -> pd.DataFrame:
    champs = df[df["is_champion"] & ~df["is_crash"]].copy()
    champs = champs.sort_values("experiment_id").reset_index(drop=True)
    champs["prev_best"] = champs["val_bpb"].shift(1)
    champs["delta"] = champs["prev_best"] - champs["val_bpb"]
    return champs


def summarize(df: pd.DataFrame) -> None:
    champs = champion_rows(df)
    valid = df[~df["is_crash"]].copy()
    print(f"Total experiments: {len(df)}")
    print(f"Valid experiments: {len(valid)}")
    print(f"Crashes: {int(df['is_crash'].sum())}")
    print(f"Champion promotions: {len(champs)}")
    mode_counts = df["mode"].value_counts()
    print("\nModes:")
    print(mode_counts.to_string())
    action_counts = df["frontier_action"].value_counts()
    print("\nFrontier actions:")
    print(action_counts.to_string())

    if champs.empty:
        print("\nNo successful champion runs yet.")
        return

    baseline = champs.iloc[0]["val_bpb"]
    best = champs["val_bpb"].min()
    best_row = champs.loc[champs["val_bpb"].idxmin()]
    print(f"\nBaseline champion val_bpb: {baseline:.6f}")
    print(f"Best champion val_bpb:     {best:.6f}")
    print(f"Total improvement:         {baseline - best:.6f} ({100 * (baseline - best) / baseline:.2f}%)")
    print(f"Best run:                  exp#{int(best_row['experiment_id'])} {best_row['description']}")


def plot_champion_progress(df: pd.DataFrame, out_path: Path) -> None:
    champs = champion_rows(df)
    valid = df[~df["is_crash"]].copy()
    if champs.empty or valid.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    frontier = valid[valid["is_frontier"]]
    archive = valid[~valid["is_frontier"] & ~valid["is_champion"]]

    if not archive.empty:
        ax.scatter(
            archive["experiment_id"],
            archive["val_bpb"],
            c="#c6ccd2",
            s=20,
            alpha=0.45,
            label="Archive",
            zorder=1,
        )
    if not frontier.empty:
        ax.scatter(
            frontier["experiment_id"],
            frontier["val_bpb"],
            c="#f4b942",
            s=42,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label="Frontier",
            zorder=3,
        )

    ax.scatter(
        champs["experiment_id"],
        champs["val_bpb"],
        c="#27ae60",
        s=72,
        edgecolors="black",
        linewidths=0.6,
        label="Champion promotions",
        zorder=4,
    )
    ax.step(
        champs["experiment_id"],
        champs["val_bpb"].cummin(),
        where="post",
        color="#1e8449",
        linewidth=2.5,
        alpha=0.85,
        label="Champion running best",
        zorder=2,
    )

    for _, row in champs.iterrows():
        desc = str(row["description"]).strip()
        if len(desc) > 44:
            desc = desc[:41] + "..."
        mode = row["mode"]
        label = f"#{int(row['experiment_id'])} {mode}: {desc}"
        ax.annotate(
            label,
            (row["experiment_id"], row["val_bpb"]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8,
            color="#145a32",
            rotation=25,
            ha="left",
            va="bottom",
        )

    baseline = champs.iloc[0]["val_bpb"]
    best = champs["val_bpb"].min()
    spread = max(baseline - best, 0.001)
    margin = spread * 0.18

    ax.set_title("Champion Progress Under Frontier Search", fontsize=14)
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_ylim(best - margin, baseline + margin)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_search_overview(df: pd.DataFrame, out_path: Path) -> None:
    valid = df[~df["is_crash"]].copy()
    if valid.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    top_ax, bottom_ax = axes

    colors = {"champion": "#27ae60", "frontier": "#f4b942", "archive": "#bdc3c7"}
    for action, color in colors.items():
        subset = valid[valid["frontier_action"].str.lower() == action]
        if subset.empty:
            continue
        top_ax.scatter(
            subset["experiment_id"],
            subset["val_bpb"],
            c=color,
            s=48 if action != "archive" else 20,
            alpha=0.85 if action != "archive" else 0.45,
            edgecolors="black" if action != "archive" else "none",
            linewidths=0.4,
            label=action.title(),
        )

    top_ax.plot(
        valid["experiment_id"],
        valid["running_best"],
        color="#1e8449",
        linewidth=2,
        alpha=0.75,
        label="Running best",
    )
    top_ax.set_ylabel("Validation BPB", fontsize=12)
    top_ax.set_title("Search Overview", fontsize=14)
    top_ax.grid(True, alpha=0.2)
    top_ax.legend(loc="upper right", fontsize=9)

    mode_order = {"baseline": 0, "mutate": 1, "recombine": 2}
    bottom_df = valid.copy()
    bottom_df["mode_y"] = bottom_df["mode"].map(mode_order).fillna(-1)
    mode_colors = {"baseline": "#5dade2", "mutate": "#7dcea0", "recombine": "#af7ac5"}
    for mode, color in mode_colors.items():
        subset = bottom_df[bottom_df["mode"] == mode]
        if subset.empty:
            continue
        bottom_ax.scatter(
            subset["experiment_id"],
            subset["mode_y"],
            c=color,
            s=60,
            alpha=0.9,
            label=mode.title(),
        )

    recomb = bottom_df[bottom_df["mode"] == "recombine"]
    for _, row in recomb.iterrows():
        bottom_ax.annotate(
            f"{row['parent']} + {row['parent_b']}",
            (row["experiment_id"], row["mode_y"]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            rotation=20,
            ha="left",
        )

    bottom_ax.set_xlabel("Experiment #", fontsize=12)
    bottom_ax.set_ylabel("Experiment mode", fontsize=12)
    bottom_ax.set_yticks([0, 1, 2], labels=["Baseline", "Mutate", "Recombine"])
    bottom_ax.grid(True, axis="x", alpha=0.2)
    bottom_ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def extract_loss_trace(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [float(match) for match in LOSS_PATTERN.findall(text)]


def plot_champion_loss_traces(df: pd.DataFrame, out_path: Path) -> None:
    champs = champion_rows(df)
    if champs.empty:
        return

    traces = []
    for _, row in champs.iterrows():
        log_path = Path(str(row["log_path"]))
        if not log_path:
            continue
        losses = extract_loss_trace(log_path)
        if not losses:
            continue
        traces.append((row, losses))

    if not traces:
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    cmap = plt.get_cmap("viridis")
    n = max(len(traces) - 1, 1)
    for idx, (row, losses) in enumerate(traces):
        color = cmap(idx / n)
        steps = list(range(len(losses)))
        label = f"#{int(row['experiment_id'])} {row['commit']} {row['description']}"
        if len(label) > 70:
            label = label[:67] + "..."
        ax.plot(steps, losses, color=color, linewidth=1.8, alpha=0.95, label=label)

    ax.set_title("Champion Training-Loss Traces", fontsize=14)
    ax.set_xlabel("Logged training step", fontsize=12)
    ax.set_ylabel("Smoothed training loss", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit("results.tsv not found")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results(RESULTS_PATH)
    summarize(df)
    plot_champion_progress(df, CHAMPION_PROGRESS_PATH)
    plot_search_overview(df, SEARCH_OVERVIEW_PATH)
    plot_champion_loss_traces(df, CHAMPION_LOSS_PATH)
    print(f"\nSaved {CHAMPION_PROGRESS_PATH}")
    print(f"Saved {SEARCH_OVERVIEW_PATH}")
    if CHAMPION_LOSS_PATH.exists():
        print(f"Saved {CHAMPION_LOSS_PATH}")
    else:
        print("Skipped champion loss traces because no saved logs were available yet.")


if __name__ == "__main__":
    main()
