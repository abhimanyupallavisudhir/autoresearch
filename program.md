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
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row shown below. The baseline will be recorded after the first run.
6. **Initialize frontier.tsv**: Create `frontier.tsv` with just the header row shown below. This tracks the active search frontier.
7. **Confirm and go**: Confirm setup looks good.

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

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 9 columns:

```
commit	parent_commit	val_bpb	delta_vs_parent	delta_vs_best	memory_gb	status	tags	description
```

1. git commit hash (short, 7 chars)
2. parent git commit hash (short, 7 chars). For the baseline, use `root`.
3. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
4. delta vs parent (`experiment - parent`) with sign, e.g. `-0.001230` — use `0.000000` for baseline/crashes
5. delta vs current best frontier commit (`experiment - best`) with sign — use `0.000000` for the first baseline row
6. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
7. status: `keep`, `candidate`, `discard`, or `crash`
8. short machine-readable tags describing the change family, separated by `|`, e.g. `opt:lr_up|arch:gelu`
9. short text description of what this experiment tried

Example:

```
commit	parent_commit	val_bpb	delta_vs_parent	delta_vs_best	memory_gb	status	tags	description
a1b2c3d	root	0.997900	0.000000	0.000000	44.0	keep	baseline	baseline
b2c3d4e	a1b2c3d	0.993200	-0.004700	-0.004700	44.2	keep	opt:lr_up	increase LR to 0.04
c3d4e5f	a1b2c3d	0.999100	+0.001200	+0.001200	44.0	candidate	act:gelu	switch to GeLU activation
d4e5f6g	b2c3d4e	0.000000	0.000000	0.000000	0.0	crash	model:width_up	double model width (OOM)
```

## Tracking the frontier

Maintain a separate `frontier.tsv` file (tab-separated, untracked by git) with this header:

```
commit	parent_commit	val_bpb	status	lookahead_remaining	tags	description
```

This table is the search state. It should stay small and actively curated.

- `status` is one of `best`, `active`, `stale`, or `dead`
- `lookahead_remaining` is an integer, usually 0, 1, or 2
- `best` is the current best-known commit by val_bpb
- `active` means the commit is still worth branching from
- `stale` means it was once active but has been pruned from regular use
- `dead` means do not branch from it again

The frontier should usually contain **3 to 8 active commits total** including the best one. Do not let it grow without bound.

Keep these kinds of commits in the frontier:

- The current best commit
- Near-miss commits that are slightly worse than best but still plausible, e.g. within roughly `0.001` to `0.003` val_bpb of best
- Diverse commits that open a meaningfully different region of design space, such as a different optimizer family, attention pattern, depth/width tradeoff, or normalization strategy

Prune these kinds of commits from the frontier:

- Clearly bad commits that are substantially worse than their parent
- Redundant commits that are basically the same idea as another stronger frontier member
- Old candidates whose follow-up experiments have failed and whose lookahead budget is exhausted

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Read `frontier.tsv` and identify:
   - the current `best` commit
   - the active frontier members
   - which tags and parent-child combinations have already been explored
2. Choose a **parent commit** to branch from. Do not always pick the best commit. Mix:
   - exploitation: branch from the best or another strong frontier member
   - exploration: branch from an underexplored active candidate
   - diversity: branch from a parent with different tags than recent experiments
3. `git checkout <parent_commit> -- train.py` if needed so `train.py` exactly matches the chosen parent before editing.
4. Design an experiment on top of that parent. Prefer one of these:
   - a direct improvement to the parent
   - a follow-up meant to unlock a previous near-miss
   - a combination experiment that applies a previously mixed-result idea on a different parent
5. Tune `train.py` with the chosen idea and commit it.
6. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
9. Record the result in `results.tsv` with parent, deltas, tags, and status. Leave `results.tsv` untracked by git.
10. Update `frontier.tsv` using the retention rules below.
11. Prune the frontier back to a small active set.
12. Continue from the next chosen parent. Do **not** immediately reset to a single surviving line of descent.

The idea is that you are a completely autonomous researcher trying things out across a small set of promising branches, not just one branch. Immediate one-step improvement is good, but not the only signal. Some changes are worth keeping alive because they unlock strong follow-up moves.

## Retention rules

Use these rules after every completed run:

1. **keep**
   - The experiment is the new best, or clearly improves on its parent with acceptable complexity.
   - Add it to `frontier.tsv` as `best` or `active`.
   - If it becomes the new best, downgrade the previous best to `active` unless it should be pruned.

2. **candidate**
   - The experiment is slightly worse than its parent or the global best, but still plausible.
   - Typical cases:
     - within roughly `0.001` to `0.003` val_bpb of the best result
     - opens a genuinely different design direction
     - seems like it needs one or two complementary follow-up changes
   - Add it to `frontier.tsv` as `active` with `lookahead_remaining` of 1 or 2.

3. **discard**
   - The experiment is clearly worse, redundant, unstable, or adds complexity without compensating gain.
   - Do not keep it in the active frontier.

4. **crash**
   - The code did not produce a valid metric.
   - Log it, learn from it, and move on.

Do not treat every non-improving run as `discard`. Near-misses are valuable search state.

## Lookahead policy

Use a small explicit lookahead budget to test path-dependent ideas.

- If a candidate is only slightly worse and the mechanism makes sense, spend up to 2 child experiments trying to make it work.
- Those child experiments should be specifically chosen to complement the candidate, not random edits.
- After each child, reduce `lookahead_remaining` by 1 unless the child itself earns `keep`.
- When `lookahead_remaining` reaches 0 and the branch still has not justified itself, mark it `stale` or `dead` and stop branching from it.

Examples:

- If a more local attention pattern loses slightly, try retuning batch size or learning rate on top of it before killing it.
- If a normalization change hurts slightly, try the initialization or residual-scaling change that logically pairs with it.
- If a model gets worse because throughput changed, try the token-budget / width / depth rebalance that fits the new regime.

## Combination policy

Actively search for interactions between edits.

- Revisit previously mixed-result tags on new parents. An idea that lost on one parent may win on another.
- If tag `Y` helped on branch A and tag `Z` helped on branch B, test `Y+Z` when they seem compatible.
- Prefer combinations whose mechanisms make sense together; do not combine changes blindly.
- When a tag has failed across multiple diverse parents, deprioritize it.

The purpose of tags is to let you reason about edit interactions explicitly instead of relying on vague memory.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (`discard` or `crash`, depending on what happened).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash`, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, revisit archived near-misses on different parents, try combinations that were previously blocked by path dependence, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
