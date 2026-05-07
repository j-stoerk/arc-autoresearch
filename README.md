# autoresearch

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real  training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is towards solving the ARC-AGI-3 benchmark. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. 

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data), and runtime utilities (dataloader, evaluation). Not modified.
- **`agent.py`** — file the agent edits. Manages the modules. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`modules`** — contains all architectural units. Agent can modify these units. **This file is edited and iterated on by the agent**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
arc-autoresearch/
├── agent.py              # manages the modules (agent modifies this)
├── prepare.py            # constants, data prep + runtime utilities (do not modify) 
├── program.md            # agent instructions (do not modify) 
├── results.tsv           ← untracked, logged each run
├── analysis.ipynb        ← adapted from analysis-3.ipynb
├── pyproject.toml        # dependencies, add arc3 env + search deps
└── modules/			   # contains all architectural units (agent modifies this)
    ├──  goal_inference.py       # GoalInference: goal/subgoal distribution, hierarchical decomposition (Step i)
    ├── hypothesis.py                 #  candidate mechanics, demo-consistency scoring, selection (Step ii)
    ├── dsl.py				# DSLConfig, operation set, weights, library (Step iii)
    ├── library.py			#  macro-operations, STITCH/DreamCoder compression, acquisition-context tags (Step iii)
    ├── search.py			# neural-guided beam + bidirectional (Step iii)
    ├── verifier.py        # symbolic constraint checking, candidate ranking (Step iv)
    ├── action.py          # Action Execution (Step v)
    ├── perception.py		# multi-repr state parser (Step vi)
    ├── memory.py			# Episodic Memory, trajectory store + retrieval (Step vii)
    ├── world_model.py		# rules, priors, MDL criterion (Step viii)
    └── loop.py		    # Belief Revision and Loop Control (Step ix)

```

## Design choices

- **Single file to modify.** The agent only touches `agent.py` and `modules`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes. Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One file, one metric.

## Platform support

This code works on either single or multiple CPUs / GPUs.


## License

MIT
