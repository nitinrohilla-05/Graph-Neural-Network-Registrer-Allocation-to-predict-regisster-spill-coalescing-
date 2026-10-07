# Graph Neural Network Register Allocation

An educational compiler and machine-learning project exploring graph neural
networks (GNNs) for register allocation, spill prediction, and move coalescing.
It includes a Python compiler pipeline, specialist prediction agents, a
confidence-weighted consensus arbiter, classical baselines, model comparisons,
and an interactive browser dashboard.

> **Project status:** This is a research and learning prototype. The main
> workflow uses synthetic compiler IR and is not a production compiler backend.

## Contents

- [Overview](#overview)
- [Models, agents, and baselines](#models-agents-and-baselines)
- [How the pipeline works](#how-the-pipeline-works)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Run tests](#run-tests)
- [CLI workflows](#cli-workflows)
- [Launch the dashboard](#launch-the-dashboard)
- [Understanding dashboard results](#understanding-dashboard-results)
- [Additional Java and ML workflow](#additional-java-and-ml-workflow)
- [Limitations](#limitations)

## Overview

Register allocation maps compiler virtual registers (program values) to a
limited set of physical registers. Two values that are live at the same time
interfere and cannot be assigned the same physical register. When the
registers are insufficient, values may be **spilled** to memory. If a `MOVE`
instruction copies one value to another, assigning both values the same
register can sometimes **coalesce** and remove that move.

The main Python workflow generates three-address code (TAC), constructs a
control-flow graph (CFG), computes liveness, and builds an interference graph.
Neural allocators and classical heuristics can then propose assignments. The
project also includes conflict-repair helpers, benchmark metrics, model
comparisons, and a static web dashboard for exploring exported results.

## Models, agents, and baselines

There are two related but distinct neural workflows:

| Component | Type | Purpose |
| --- | --- | --- |
| **R-GCN** | Relational GNN architecture and specialist agent | Relational message passing; in the agent workflow, focuses on move chains and affinity edges |
| **R-GAT** | Relational GNN architecture | Attention-based message passing over graph relations |
| **R-SAGE** | Relational GNN architecture | Mean neighborhood aggregation over graph relations |
| **R-GIN** | Relational GNN architecture | Relation-aware sum aggregation and MLP updates |
| **GAT** | Specialist agent architecture | Looks at attention and neighborhood pressure |
| **GraphSAGE** | Specialist agent architecture | Looks at local neighborhoods, hubs, and dense subgraphs |
| **GCN** | Specialist agent architecture | Looks at global interference pressure and loop-hot variables |

The four relational architectures are **R-GCN, R-GAT, R-SAGE, and R-GIN**.
The four specialist agents are **R-GCN, GAT, GraphSAGE, and GCN**. These are
different groupings for training and comparison; some names overlap.

Other dashboard strategies are not additional GNN architectures:

- **Consensus Arbiter** combines the specialist agents' confidence-weighted
  predictions. It is an ensemble decision-maker, not a separately trained GNN.
- **Chaitin-Briggs** is a classical graph-coloring heuristic and serves as a
  non-AI baseline.
- **Random** is a simple randomized baseline.

Which model buttons appear in the dashboard depends on the dataset loaded.

## How the pipeline works

1. Generate a TAC program and derive its basic blocks and control flow.
2. Compute liveness and construct interference and move/coalescing edges.
3. Run the selected GNN model or the specialist agents on the graph.
4. Optionally combine agent predictions with confidence-weighted consensus.
5. Repair invalid register conflicts where configured and compare with
   classical baselines.
6. Export assignments and diagnostics for evaluation or dashboard display.

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Entry point for training, evaluation, generation, and export |
| `compiler/` | TAC IR, CFG, liveness, interference graphs, and Chaitin-Briggs allocation |
| `dataset/` | Synthetic program generation and graph datasets |
| `agents/` | Specialist agents, checkpoint registry, and consensus arbiter |
| `models/` | Relational GNN architectures, training, and allocation repair |
| `evaluation/` | Benchmarks, metrics, comparisons, and feature ablations |
| `tests/` | Python test suite |
| `web/` | Static interactive dashboard and exported JSON datasets |
| `gnn-regalloc/` | Separate Java compiler and Python ML workflow |
| `report/`, `docs/`, `results/` | Reports, baseline notes, and generated results |

The root `requirements.txt` belongs to the main Python project. The
`gnn-regalloc/requirements.txt` file belongs to the separate nested workflow.

## Setup

Use Python from the repository root. A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate in a POSIX shell:

```bash
source .venv/bin/activate
```

Install the root project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include PyTorch, NetworkX, Matplotlib, scikit-learn, NumPy, and
pytest. PyTorch installation can vary by operating system and accelerator; use
the official PyTorch installation selector if the default package is not
appropriate for your machine.

## Run tests

Run all tests from the repository root:

```bash
python -m pytest
```

Run one focused test module:

```bash
python -m pytest tests/test_agents.py
```

Passing tests verify the tested project behavior; they do not establish that
the allocator will improve performance on real-world compiler workloads.

## CLI workflows

Use `python main.py --help` to see available options. The most commonly used
commands are below.

### Train a relational architecture

```bash
python main.py --mode train --model rgcn --epochs 20 --samples 60 --registers 4
```

Choose `rgcn`, `rgat`, `rsage`, or `gin` with `--model`. By default, training
writes the checkpoint to `gnn_allocator.pt`; use `--output` to select a
different location.

### Train and evaluate specialist agents

Quick CPU smoke run:

```bash
python main.py --mode train-agents --quick --checkpoint-dir checkpoints
python main.py --mode evaluate-agents --quick
```

Custom training run:

```bash
python main.py --mode train-agents --samples 60 --epochs 20 --registers 4 --checkpoint-dir checkpoints
```

Generate an agent report:

```bash
python main.py --mode agents-report --quick
```

The report workflow writes files such as `results/agents_results.json` and
`results/agents_results.md`.

### Compare models

Compare the architectures and baselines:

```bash
python main.py --mode compare-models --registers 4
```

Aggregate a comparison over multiple generated programs:

```bash
python main.py --mode compare-models --samples 20 --registers 4
```

The comparison workflow exports `web/comparison_data.json` and writes a
comparison plot under `report/`.

### Export graph data

Export the multi-agent dashboard dataset:

```bash
python main.py --mode export-web-data --registers 4
```

Export raw synthetic graphs:

```bash
python main.py --mode export-json --samples 60 --registers 4
```

### Available modes

| Mode | Purpose |
| --- | --- |
| `train` | Train one relational GNN architecture |
| `evaluate` | Benchmark a model checkpoint against allocator baselines |
| `run-compiler` | Run the generated compiler allocation demonstration |
| `export-web-data` | Export multi-agent graph data for the dashboard |
| `export-json` | Export generated graph samples |
| `model-comparison` | Run multi-seed architecture comparison |
| `ablation` | Evaluate feature ablations |
| `train-agents` | Train the specialist-agent checkpoints |
| `evaluate-agents` | Evaluate agents and consensus |
| `agents-report` | Generate agent result files |
| `compare-models` | Compare relational architectures and baselines |

Important options include `--registers` (default `4`), `--samples` (default
`60`), `--epochs` (default `20`), `--hidden-dim` (default `64`), `--seed`
(default `42`), `--model`, `--agent`, `--checkpoint-dir`, and `--model-path`.
Keep program data, seed, register count, checkpoints, and repair behavior
consistent when comparing results.

## Launch the dashboard

The dashboard is a static web app in `web/`. Start a local server at the
repository root:

```bash
python -m http.server 8000 --directory web
```

Open [http://localhost:8000](http://localhost:8000) in a browser. The checked-in
JSON files provide example data. To display a fresh multi-agent run, export
the web data first, then refresh the page.

| File | Dashboard data |
| --- | --- |
| `web/data.json` | Multi-agent graph, assignments, and diagnostics |
| `web/comparison_data.json` | Cross-architecture comparison graph and metrics |
| `web/embedded_data.js` | Embedded datasets used for file/offline rendering |

Use **Load Comparison Run** to switch between the multi-agent and
architecture-comparison datasets.

## Understanding dashboard results

- **Graph nodes** represent virtual registers. Interference edges connect
  values that cannot share a register; coalescing edges represent move
  relationships.
- **Node colors** show the selected allocator's register assignment or spill
  decision.
- **Agent Patterns** displays available specialist diagnostics and a summary
  for the selected allocator.
- **Agreement Matrix** compares the selected allocator with others and may
  also show dataset-level pairwise matrices. Exact register agreement is
  stricter than agreement on spill versus non-spill decisions.
- **TAC IR** shows the program instructions and the selected allocator's
  per-variable assignment summary.
- **CFG** shows basic blocks and their successors when that dataset includes
  CFG information.

Agreement measures whether two allocators made the same decisions, not
whether those decisions are optimal. A low spill count alone does not prove
that an allocation is valid or faster; consider conflict validity, spill cost,
move elimination, and evaluation conditions together.

## Additional Java and ML workflow

`gnn-regalloc/` is a separate workflow with Java compiler-side components and
a Python/PyTorch-geometric ML side. Its setup and reproduction steps are
documented in [`gnn-regalloc/README.md`](gnn-regalloc/README.md). Use its
nested `requirements.txt` only when working on that project.

## Limitations

- The main generator uses synthetic TAC programs; behavior may not generalize
  to real-world compiler workloads.
- Results depend on the generated program, training data, checkpoint,
  hyperparameters, register count, and conflict-repair settings.
- Chaitin-Briggs and Random are baselines, not neural architectures.
- Dashboard JSON is an exported snapshot and only changes when it is
  regenerated or another dataset is selected.
- The nested `gnn-regalloc/` project is a separate workflow and should not be
  confused with the root Python CLI.

For implementation and evaluation details, see [`report/`](report/),
[`docs/baseline_results.md`](docs/baseline_results.md), and
[`gnn-regalloc/report/`](gnn-regalloc/report/). The more detailed companion
guide is [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
