# autoresearch

This is an experiment to have an AI agent autonomously improve a neuro-symbolic agent
that solves interactive grid environments without instructions, goals, or explicit rules.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `agent.py` — the file you modify. Imports all modules, wires the AgentLoop, sets all hyperparameters.
   - `modules/` — architectural modules. You may modify these if a structural change is necessary, but prefer modifying agent.py configuration first.
4. **Verify environments exist**: Check that `.cache/arc3/` contains the 25 public task files. If not, tell the human to run uv run `prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment check the hardware, and runs on one or multiple CPUs / GPUs. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run agent.py`.

**What you CAN do**
- Modify `agent.py` — this is the primary file you edit. Everything is fair game:
DSL operation weights, beam width, hypothesis scoring function, MDL threshold,
memory retrieval strategy (recency / relevance / salience weights), world model
rule types, goal inference prior, loop control conditions.
- Modify files in `modules/` if a structural change to a module is necessary for
a meaningful architectural improvement. Keep changes minimal and focused.

**What you CANNOT do**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness,
environment loader, episode iterator, action budget constants, and evaluate_rhae().
- Modify `evaluate_rhae()`. It is the ground-truth metric.
- Exceed the action budget. The 5× human action count hard limit is enforced by the environment.
- Install new packages or add dependencies beyond what is in `pyproject.toml`.

ARC-AGI-3 uses an interactive 64×64 color-grid environments.
No instructions, no stated goals, no explicit rules are provided. The agent observes the grid,
takes discrete actions, and must infer goals, world mechanics, and win conditions entirely on its own.

Scoring uses the Relative Human Action Efficiency (RHAE), penalized by a power-law:
A score of 1.0 means the agent matched the human baseline action count exactly.
Taking 10× more actions than the human baseline yields approximately 1% credit (exponent = 2).
An automatic score of 0 is assigned if the agent exceeds 5× the human action count on any level.

Levels are weighted: Level 1 → weight 1, Level 2 → weight 2, etc.
The goal is simple: maximize the weighted average RHAE across all 25 public environments.

**The goal is simple: get the highest RHAE score.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful RHAE gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 RHAE improvement that adds 20 lines of hacky code? Probably not worth it. A RHAE improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.


## Architecture Overview
The agent runs a closed inference loop over each environment episode:

Step	Module	Description
Init	agent.py	Perceive initial state, instantiate world model
(i)	modules/goal_inference.py	Infer goals/subgoals from limited observations
(ii)	modules/hypothesis.py	Form and rank hypotheses about game mechanics
(iii)	modules/dsl.py + modules/library.py + modules/search.py	World-model-conditioned plan search
(iv)	modules/verifier.py	Symbolic constraint checking, candidate ranking
(v)	agent.py → env	Execute verified action
(vi)	modules/perception.py	Parse resulting state via multiple representations
(vii)	modules/memory.py	Store trajectory, retrieve analogical episodes
(viii)	modules/world_model.py	Revise rules, update DSL registry via MDL
(ix)	modules/loop.py	Belief revision, loop control, termination
The world model (Step viii) feeds back directly into the DSL (Step iii): after each world model
update, the operation registry is recomputed, and only operations whose preconditions are consistent
with the current mechanic hypothesis are active. This is the core design principle — the agent
does not overcome the search space by brute force, but by shrinking it through a learned vocabulary.


## Output format
Once the script finishes it prints a summary like:

text
---
rhae_score       0.312000
avg_actions      14.3
total_seconds    318.4
episodes_run     25
goal_inf_acc     0.71
hyp_acc          0.84
plan_depth_mean  4.2
Extract the key metric:

bash `grep rhae_score run.log`
**Logging Results**: Log each experiment to results.tsv (tab-separated, NOT comma-separated — commas break in descriptions).
The TSV has a header row and 5 columns:

'''
commit    rhae_score    avg_actions    status    description
commit — git commit hash, short (7 chars)

rhae_score — achieved score (e.g. 0.312000); use 0.000000 for crashes

avg_actions — mean actions per episode, round to .1f; use 0.0 for crashes

status — keep, discard, or crash

description — short text description of what this experiment tried
'''

Example:
'''
commit    rhae_score    avg_actions    status    description
a1b2c3d   0.312000      14.3          keep      baseline
b2c3d4e   0.341000      12.1          keep      increase beam width 3→5, MDL threshold 0.1→0.08
c3d4e5f   0.298000      18.7          discard   disable library learning, pure primitive search
d4e5f6g   0.000000      0.0           crash     bidirectional synthesis OOM on env 17
'''

Do NOT commit results.tsv. Leave it untracked by git.

## The Experiment Loop
The experiment runs on a dedicated branch (e.g. autoresearch/may7).

LOOP FOREVER:
1. Look at the git state — what branch and commit are you on?
2. Tune agent.py (or a module if necessary) with an experimental idea.
3. git commit
4. Run the experiment: bash, uv run agent.py > run.log 2>&1
Redirect everything. Do NOT use tee or let output flood your context.
5. Read out the results: bash, grep "rhae_score\|avg_actions" run.log
6. If the grep output is empty, the run crashed. Run tail -n 50 run.log to read the
Python stack trace and attempt a fix. If you cannot get things working after a few
attempts, give up: log crash and move on.
7. Record the result in results.tsv. Do NOT commit the TSV.
8. If rhae_score improved (higher), advance the branch — keep the git commit.
9. If rhae_score is equal or worse, git reset --hard HEAD~1 back to where you started.

**Timeout**: Each experiment should complete within TIME_BUDGET seconds plus a few seconds
of startup overhead. If a run exceeds 2 × TIME_BUDGET, kill it and treat as failure.
**Crashes**: Use your judgment. A typo or missing import → fix and re-run. A fundamentally
broken idea (e.g. search OOM, invalid DSL precondition type) → log crash, revert, move on.

NEVER STOP
Once the experiment loop has begun, do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human may be away and expects you to continue working indefinitely until manually stopped.
You are autonomous.

If you run out of ideas, think harder:
- Re-read modules/world_model.py — is the MDL criterion tight enough? Too tight?
- Re-read modules/search.py — is the beam width optimal? Is bidirectional synthesis helping?
- Re-read modules/hypothesis.py — is demo-consistency scoring the right selection criterion?
- Re-read modules/memory.py — is the recency/relevance/salience balance right?
- Try combining previous near-misses.
- Try more radical changes: a different goal inference prior, a different loop control policy,
disabling library learning to test if it helps or hurts, changing the MDL penalty weight.

The loop runs until the human interrupts you, period.

## Ideas to Explore
This list is a starting point, not a constraint. Use it as inspiration, add your own.

1. Criticality: Systems that undergo a rapid or discontinuous change, such
as a phase transition in a control variable, where a new organization lends
itself to a new description.
2. Compression: Compressed representations internal to the model are exploited by the model to increase the representation’s fidelity or efficiency.
3. Novel Bases: New bases or functions are discovered that provide an internal “alphabet” that is used to encode regularities.
4. Generalization: In adaptive systems the capabilities arising from one task
can be used to solve different tasks, enabling competence arising from the
union of discrete performance.

Other tweaks:

DSL / Search (Step iii)
- Try disabling bidirectional synthesis and compare against unidirectional
- Try disabling library learning entirely (pure primitive search) to measure its contribution

Hypothesis (Step ii)
- Try entropy-based selection instead of MAP selection for the operative hypothesis

World Model (Step viii)
- Try geometric vs. physics vs. objectness priors independently to isolate contribution

Memory (Step vii)
- Try disabling cross-episode transfer to measure generalization contribution

Goal Inference (Step i)
- Try top-1 vs. top-3 goal candidates and measure downstream impact