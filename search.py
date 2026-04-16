"""
Path-aware experiment orchestration for autoresearch.

This script does not edit train.py itself. It manages search state around the
LLM-driven experiment loop so the agent can keep a frontier of promising code
states instead of greedily following a single incumbent.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_DIR = Path(".autoresearch")
STATE_PATH = STATE_DIR / "state.json"
RESULTS_PATH = Path("results.tsv")
CONFIG_PATTERN = re.compile(r"^[+-]\s*([A-Z][A-Z0-9_]{1,})\s*=")
IDENT_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_git(args: list[str], capture_output: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=capture_output,
        text=True,
    )
    return proc.stdout.strip() if capture_output else ""


def git_short_commit(rev: str = "HEAD") -> str:
    return run_git(["rev-parse", "--short=7", rev])


def git_full_commit(rev: str = "HEAD") -> str:
    return run_git(["rev-parse", rev])


def git_current_branch() -> str:
    return run_git(["branch", "--show-current"])


def git_commit_exists(commit: str) -> bool:
    proc = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"])
    return proc.returncode == 0


def git_diff_train(parent: str, commit: str) -> str:
    return run_git(["diff", f"{parent}..{commit}", "--", "train.py"])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_results_header(path: Path) -> None:
    ensure_parent(path)
    if path.exists() and path.stat().st_size > 0:
        return
    path.write_text(
        "experiment_id\ttimestamp\tmode\tcommit\tparent\tparent_b\tval_bpb\tmemory_gb\tstatus\tdescription\ttags\tfrontier_action\tlog_path\n",
        encoding="utf-8",
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"State file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_tags_from_diff(diff: str) -> list[str]:
    tags: set[str] = set()
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        cfg = CONFIG_PATTERN.match(line)
        if cfg:
            tags.add(cfg.group(1).lower())
        for ident in IDENT_PATTERN.findall(line):
            if ident.isupper():
                continue
            if ident in {"self", "True", "False", "None"}:
                continue
            tags.add(ident.lower())
    return sorted(tags)


def relative_gap(value: float, best: float) -> float:
    if best <= 0:
        return 0.0
    return max(0.0, (value - best) / best)


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def compute_novelty(tags: set[str], frontier_records: list[dict[str, Any]]) -> float:
    if not frontier_records:
        return 1.0
    return min(jaccard_distance(tags, set(rec.get("tags", []))) for rec in frontier_records)


def compute_synergy(
    tags: set[str],
    experiments: dict[str, dict[str, Any]],
    exclude_commit: str | None = None,
) -> float:
    if not tags:
        return 0.0
    pair_scores = []
    for exp in experiments.values():
        if exclude_commit is not None and exp["commit"] == exclude_commit:
            continue
        if exp.get("status") == "crash":
            continue
        exp_tags = set(exp.get("tags", []))
        overlap = len(tags & exp_tags)
        if overlap == 0:
            continue
        score = max(0.0, 1.0 - exp["relative_gap"])
        pair_scores.append(overlap * score)
    if not pair_scores:
        return 0.0
    return min(1.0, sum(pair_scores) / (len(tags) * max(1, len(pair_scores))))


def choose_frontier(
    experiments: dict[str, dict[str, Any]],
    frontier_size: int,
    epsilon_rel: float,
    novelty_floor: float,
    novelty_weight: float,
) -> tuple[list[str], str | None]:
    viable = [
        exp for exp in experiments.values()
        if exp.get("status") != "crash" and exp.get("val_bpb", 0.0) > 0.0
    ]
    if not viable:
        return [], None
    best = min(viable, key=lambda exp: exp["val_bpb"])
    champion = best["commit"]
    frontier = [champion]
    frontier_records = [best]
    candidates = sorted(
        (exp for exp in viable if exp["commit"] != champion),
        key=lambda exp: (exp["relative_gap"], exp["memory_gb"]),
    )
    for exp in candidates:
        if len(frontier) >= frontier_size:
            break
        tags = set(exp.get("tags", []))
        novelty = compute_novelty(tags, frontier_records)
        exp["novelty"] = novelty
        if exp["relative_gap"] <= epsilon_rel or novelty >= novelty_floor:
            frontier.append(exp["commit"])
            frontier_records.append(exp)
    if len(frontier) < min(frontier_size, len(viable)):
        remaining = [exp for exp in candidates if exp["commit"] not in frontier]
        ranked = sorted(
            remaining,
            key=lambda exp: exp["relative_gap"] - novelty_weight * exp.get("novelty", 0.0),
        )
        for exp in ranked:
            if len(frontier) >= frontier_size:
                break
            frontier.append(exp["commit"])
    return frontier, champion


def frontier_action(commit: str, frontier: list[str], champion: str | None) -> str:
    if commit == champion:
        return "champion"
    if commit in frontier:
        return "frontier"
    return "archive"


def append_result_row(path: Path, row: list[str]) -> None:
    ensure_results_header(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def summarize_frontier(state: dict[str, Any]) -> str:
    experiments = state["experiments"]
    lines = []
    champion = state.get("champion")
    for idx, commit in enumerate(state.get("frontier", []), start=1):
        exp = experiments[commit]
        role = "champion" if commit == champion else "frontier"
        lines.append(
            f"{idx}. {commit} {role} val_bpb={exp['val_bpb']:.6f} "
            f"gap={100 * exp['relative_gap']:.3f}% tags={','.join(exp.get('tags', [])[:6])}"
        )
    return "\n".join(lines)


def init_command(args: argparse.Namespace) -> None:
    state_path = Path(args.state)
    results_path = Path(args.results)
    baseline_commit = git_short_commit(args.baseline_commit)
    baseline_full = git_full_commit(args.baseline_commit)
    state = {
        "version": 2,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "run_tag": args.run_tag,
        "baseline_branch": git_current_branch(),
        "baseline_commit": baseline_commit,
        "baseline_commit_full": baseline_full,
        "frontier_size": args.frontier_size,
        "epsilon_rel": args.epsilon_rel,
        "novelty_floor": args.novelty_floor,
        "novelty_weight": args.novelty_weight,
        "recombine_every": args.recombine_every,
        "mutate_champion_prob": args.mutate_champion_prob,
        "results_path": str(results_path),
        "experiment_counter": 0,
        "champion": None,
        "frontier": [],
        "experiments": {},
    }
    save_state(state_path, state)
    ensure_results_header(results_path)
    print(f"Initialized search state at {state_path}")
    print(f"Baseline branch: {state['baseline_branch']}")
    print(f"Baseline commit: {baseline_commit}")
    print(f"Results file: {results_path}")


def register_command(args: argparse.Namespace) -> None:
    state_path = Path(args.state)
    state = load_state(state_path)
    results_path = Path(state.get("results_path", args.results))
    commit = git_short_commit(args.commit)
    parent = git_short_commit(args.parent)
    if not git_commit_exists(commit):
        raise SystemExit(f"Unknown commit: {commit}")
    if not git_commit_exists(parent):
        raise SystemExit(f"Unknown parent commit: {parent}")

    val_bpb = float(args.val_bpb)
    memory_gb = float(args.memory_gb)
    status = args.status
    diff = git_diff_train(parent, commit)
    auto_tags = extract_tags_from_diff(diff)
    tags = sorted(set(auto_tags) | set(args.tags))

    prior_successes = [
        exp["val_bpb"]
        for exp in state["experiments"].values()
        if exp.get("status") != "crash" and exp.get("val_bpb", 0.0) > 0.0
    ]
    best_before = min(prior_successes) if prior_successes else val_bpb
    rel_gap = 0.0 if status == "crash" else relative_gap(val_bpb, best_before)

    record = {
        "experiment_id": state["experiment_counter"] + 1,
        "commit": commit,
        "parent": parent,
        "parent_b": args.parent_b or "",
        "timestamp": utc_now(),
        "mode": args.mode,
        "description": args.description,
        "status": status,
        "val_bpb": 0.0 if status == "crash" else val_bpb,
        "memory_gb": 0.0 if status == "crash" else memory_gb,
        "tags": tags,
        "auto_tags": auto_tags,
        "relative_gap": 999.0 if status == "crash" else rel_gap,
        "novelty": 0.0,
        "synergy": 0.0,
        "log_path": args.log_path or "",
    }
    state["experiments"][commit] = record

    frontier, champion = choose_frontier(
        experiments=state["experiments"],
        frontier_size=state["frontier_size"],
        epsilon_rel=state["epsilon_rel"],
        novelty_floor=state["novelty_floor"],
        novelty_weight=state["novelty_weight"],
    )
    frontier_records = [state["experiments"][c] for c in frontier if c != commit]
    record["novelty"] = compute_novelty(set(tags), frontier_records) if status != "crash" else 0.0
    record["synergy"] = (
        compute_synergy(set(tags), state["experiments"], exclude_commit=commit)
        if status != "crash" else 0.0
    )

    frontier, champion = choose_frontier(
        experiments=state["experiments"],
        frontier_size=state["frontier_size"],
        epsilon_rel=state["epsilon_rel"],
        novelty_floor=state["novelty_floor"],
        novelty_weight=state["novelty_weight"],
    )
    state["frontier"] = frontier
    state["champion"] = champion
    state["experiment_counter"] += 1
    state["updated_at"] = utc_now()
    action = frontier_action(commit, frontier, champion)
    save_state(state_path, state)

    append_result_row(
        results_path,
        [
            str(record["experiment_id"]),
            record["timestamp"],
            record["mode"],
            commit,
            parent,
            record["parent_b"],
            f"{record['val_bpb']:.6f}",
            f"{record['memory_gb']:.1f}",
            status,
            args.description,
            ",".join(tags),
            action,
            record["log_path"],
        ],
    )
    print(
        f"Recorded exp#{record['experiment_id']} {commit}: status={status} val_bpb={record['val_bpb']:.6f} "
        f"novelty={record['novelty']:.3f} synergy={record['synergy']:.3f} action={action}"
    )
    if frontier:
        print("Current frontier:")
        print(summarize_frontier(state))


def score_parent(
    exp: dict[str, Any],
    champion: str | None,
    mutate_champion_prob: float,
) -> float:
    base = 1.0 - min(exp["relative_gap"], 1.0)
    novelty_bonus = 0.2 * exp.get("novelty", 0.0)
    synergy_bonus = 0.2 * exp.get("synergy", 0.0)
    champion_bonus = mutate_champion_prob if exp["commit"] == champion else 0.0
    return max(0.0, base + novelty_bonus + synergy_bonus + champion_bonus)


def suggest_recombine(state: dict[str, Any]) -> str:
    frontier = state.get("frontier", [])
    experiments = state["experiments"]
    if len(frontier) < 2:
        return "mode=mutate\nreason=not_enough_frontier_entries_for_recombination"
    pairs: list[tuple[float, str, str]] = []
    for i, a in enumerate(frontier):
        for b in frontier[i + 1:]:
            exp_a = experiments[a]
            exp_b = experiments[b]
            tags_a = set(exp_a.get("tags", []))
            tags_b = set(exp_b.get("tags", []))
            overlap = len(tags_a & tags_b)
            union = len(tags_a | tags_b) or 1
            complementarity = 1.0 - (overlap / union)
            strength = (1.0 - exp_a["relative_gap"]) + (1.0 - exp_b["relative_gap"])
            synergy = exp_a.get("synergy", 0.0) + exp_b.get("synergy", 0.0)
            score = complementarity + 0.5 * strength + 0.5 * synergy
            pairs.append((score, a, b))
    score, a, b = max(pairs)
    tags_a = ",".join(experiments[a].get("tags", [])[:8])
    tags_b = ",".join(experiments[b].get("tags", [])[:8])
    return (
        "mode=recombine\n"
        f"parent_a={a}\n"
        f"parent_b={b}\n"
        f"score={score:.3f}\n"
        f"reason=high_complementarity_on_frontier\n"
        f"tags_a={tags_a}\n"
        f"tags_b={tags_b}"
    )


def suggest_mutation(state: dict[str, Any]) -> str:
    frontier = state.get("frontier", [])
    experiments = state["experiments"]
    if not frontier:
        baseline = state["baseline_commit"]
        return f"mode=mutate\nparent={baseline}\nreason=no_successful_experiments_yet"
    champion = state.get("champion")
    weighted = []
    for commit in frontier:
        exp = experiments[commit]
        score = score_parent(exp, champion, state["mutate_champion_prob"])
        weighted.append((score, commit))
    total = sum(score for score, _ in weighted)
    if total <= 0:
        commit = frontier[0]
    else:
        threshold = random.random() * total
        running = 0.0
        commit = frontier[0]
        for score, candidate in weighted:
            running += score
            if running >= threshold:
                commit = candidate
                break
    exp = experiments[commit]
    return (
        "mode=mutate\n"
        f"parent={commit}\n"
        f"reason=frontier_sampling\n"
        f"parent_val_bpb={exp['val_bpb']:.6f}\n"
        f"parent_gap_percent={100 * exp['relative_gap']:.3f}\n"
        f"parent_tags={','.join(exp.get('tags', [])[:8])}"
    )


def suggest_command(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state))
    if args.seed is not None:
        random.seed(args.seed)
    use_recombine = (
        args.mode == "recombine"
        or (
            args.mode == "auto"
            and state["experiment_counter"] > 0
            and state["experiment_counter"] % state["recombine_every"] == 0
        )
    )
    if use_recombine:
        print(suggest_recombine(state))
    else:
        print(suggest_mutation(state))


def frontier_command(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state))
    if not state.get("frontier"):
        print("Frontier is empty.")
        return
    print(summarize_frontier(state))


def history_command(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state))
    experiments = sorted(
        state["experiments"].values(),
        key=lambda exp: exp["timestamp"],
    )
    limit = args.limit if args.limit is not None else len(experiments)
    for exp in experiments[-limit:]:
        print(
            f"exp#{exp.get('experiment_id', '?')} {exp['timestamp']} {exp['commit']} "
            f"mode={exp.get('mode', 'mutate')} parent={exp['parent']} "
            f"parent_b={exp.get('parent_b', '')} status={exp['status']} val_bpb={exp['val_bpb']:.6f} "
            f"gap={100 * exp['relative_gap']:.3f}% tags={','.join(exp.get('tags', [])[:8])} "
            f"desc={exp['description']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_p = subparsers.add_parser("init", help="initialize search state")
    init_p.add_argument("--run-tag", required=True)
    init_p.add_argument("--baseline-commit", default="HEAD")
    init_p.add_argument("--state", default=str(STATE_PATH))
    init_p.add_argument("--results", default=str(RESULTS_PATH))
    init_p.add_argument("--frontier-size", type=int, default=5)
    init_p.add_argument("--epsilon-rel", type=float, default=0.004)
    init_p.add_argument("--novelty-floor", type=float, default=0.55)
    init_p.add_argument("--novelty-weight", type=float, default=0.15)
    init_p.add_argument("--recombine-every", type=int, default=10)
    init_p.add_argument("--mutate-champion-prob", type=float, default=0.35)
    init_p.set_defaults(func=init_command)

    register_p = subparsers.add_parser("register", help="record experiment outcome")
    register_p.add_argument("--commit", default="HEAD")
    register_p.add_argument("--parent", required=True)
    register_p.add_argument("--val-bpb", type=float, required=True)
    register_p.add_argument("--memory-gb", type=float, required=True)
    register_p.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    register_p.add_argument("--mode", choices=["baseline", "mutate", "recombine"], default="mutate")
    register_p.add_argument("--description", required=True)
    register_p.add_argument("--parent-b", default="")
    register_p.add_argument("--tags", nargs="*", default=[])
    register_p.add_argument("--log-path", default="")
    register_p.add_argument("--state", default=str(STATE_PATH))
    register_p.add_argument("--results", default=str(RESULTS_PATH))
    register_p.set_defaults(func=register_command)

    suggest_p = subparsers.add_parser("suggest", help="suggest next parent or recombination")
    suggest_p.add_argument("--state", default=str(STATE_PATH))
    suggest_p.add_argument("--mode", choices=["auto", "mutate", "recombine"], default="auto")
    suggest_p.add_argument("--seed", type=int)
    suggest_p.set_defaults(func=suggest_command)

    frontier_p = subparsers.add_parser("frontier", help="show the current frontier")
    frontier_p.add_argument("--state", default=str(STATE_PATH))
    frontier_p.set_defaults(func=frontier_command)

    history_p = subparsers.add_parser("history", help="show recent experiments")
    history_p.add_argument("--state", default=str(STATE_PATH))
    history_p.add_argument("--limit", type=int, default=10)
    history_p.set_defaults(func=history_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
