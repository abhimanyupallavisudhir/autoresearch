# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `search.py` — search-state orchestrator. Use this to manage the frontier and results.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize search state**: Run `uv run search.py init --run-tag <tag>`. This creates `.autoresearch/state.json` and `results.tsv`.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Search strategy

Do not do pure greedy hill climbing from a single incumbent. This repo is path-dependent, so a change that looks weak on one parent can be a strong stepping stone on another.

Use a **population-based search**:

- Keep a **frontier** of up to 5 commits: 1 champion plus up to 4 diverse runners-up.
- Mutate from any frontier member, not just the champion.
- Keep near-best commits if they are structurally different enough to be plausible stepping stones.
- Every 10 experiments, attempt one **recombination** experiment by combining two complementary frontier branches.
- Treat crashes as failures, but do not collapse the frontier because one branch failed.
- For especially promising candidates, plan to re-run later for robustness; do not promote or demote solely on one noisy lucky run if the gain is tiny.

The `search.py` script manages the frontier, diversity heuristics, and parent/recombination suggestions. Use it instead of manually deciding every step.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 13 columns:

```
experiment_id	timestamp	mode	commit	parent	parent_b	val_bpb	memory_gb	status	description	tags	frontier_action	log_path
```

1. `experiment_id`: sequential integer assigned by the orchestrator
2. `timestamp`: UTC timestamp when the experiment was registered
3. `mode`: `baseline`, `mutate`, or `recombine`
4. `commit`: git commit hash (short, 7 chars)
5. `parent`: primary parent commit hash the experiment branched from
6. `parent_b`: optional second parent for recombination experiments
7. `val_bpb`: achieved value (e.g. 1.234567) — use 0.000000 for crashes
8. `memory_gb`: peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
9. `status`: `keep`, `discard`, or `crash`
10. `description`: short text description of what this experiment tried
11. `tags`: optional tags; `search.py` can also infer tags from the diff
12. `frontier_action`: one of `champion`, `frontier`, or `archive`
13. `log_path`: path to the saved log file for this run

Example:

```
experiment_id	timestamp	mode	commit	parent	parent_b	val_bpb	memory_gb	status	description	tags	frontier_action	log_path
1	2026-04-16T12:00:00+00:00	baseline	a1b2c3d	a1b2c3d		0.997900	44.0	keep	baseline	baseline	champion	logs/001-baseline.log
2	2026-04-16T12:07:00+00:00	mutate	b2c3d4e	a1b2c3d		0.993200	44.2	keep	increase LR to 0.04	matrix_lr,lr	champion	logs/002-lr.log
3	2026-04-16T12:14:00+00:00	mutate	c3d4e5f	a1b2c3d		0.995000	44.0	discard	switch schedule	warmdown_ratio,scheduler	frontier	logs/003-schedule.log
4	2026-04-16T12:21:00+00:00	recombine	d4e5f6g	b2c3d4e	c3d4e5f	0.000000	0.0	crash	combine LR and schedule	depth,model_dim	archive	logs/004-recombine.log
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Ask the orchestrator what to try next:
   - `uv run search.py suggest`
   - If it returns `mode=mutate`, start from `parent=<commit>`.
   - If it returns `mode=recombine`, create a new commit that intentionally combines the main ideas from `parent_a` and `parent_b`.
2. Check out the chosen parent commit into the working tree:
   - `git checkout <parent> -- train.py`
   - For recombination, inspect both parent diffs before editing.
3. Tune `train.py` with one clear experimental idea. Keep the description crisp enough that it can become a useful log entry later.
4. `git commit` the candidate change.
5. Save each run to its own log file, never overwrite a single shared `run.log`. Example:
   - `mkdir -p logs`
   - `log_path="logs/$(date -u +%Y%m%dT%H%M%SZ)-candidate.log"`
   - `uv run train.py > "$log_path" 2>&1`
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" "$log_path"`
7. If the grep output is empty, the run crashed. Run `tail -n 50 "$log_path"` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, log a crash and move on.
8. Register the result with the orchestrator. Example:
   - `uv run search.py register --mode mutate --parent <parent> --val-bpb 0.993200 --memory-gb 44.2 --status keep --description "increase LR to 0.04" --tags lr matrix_lr --log-path "$log_path"`
   - For recombinations, also pass `--parent-b <parent_b>`.
   - For crashes, use `--val-bpb 0 --memory-gb 0 --status crash`.
9. Use `uv run search.py frontier` to inspect the active frontier when needed.
10. Do not collapse back to a single best branch. The frontier is the search state. The working tree can move around freely as long as commits are logged and the frontier remains intact.

The idea is that you are a completely autonomous researcher trying things out across multiple nearby basins, not just greedily marching forward from a single line of descent. Favor edits that are understandable, composable, and easy to recombine.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
