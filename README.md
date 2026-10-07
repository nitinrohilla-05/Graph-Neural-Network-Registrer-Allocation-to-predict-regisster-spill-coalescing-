# Graph Neural Network Register Allocation

An educational compiler and machine-learning project exploring graph neural
networks (GNNs) for register allocation, spill prediction, and move coalescing.
It includes a Python compiler pipeline, four specialized allocation agents, a
confidence-weighted consensus arbiter, classical baselines, model comparisons,
and an interactive browser dashboard.

> **Project guide:** See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for setup,
> architecture, model explanations, commands, dashboard instructions, and
> project limitations.

## At a glance

- **Compiler pipeline:** Generates three-address code (TAC), builds control
  flow and interference graphs, and extracts graph features.
- **Four specialist agents:** R-GCN, GAT, GraphSAGE, and GCN make predictions
  focused on different graph patterns.
- **Four relational architectures:** R-GCN, R-GAT, R-SAGE, and R-GIN can be
  trained and compared.
- **Consensus Arbiter:** Combines the specialist agents' predictions using
  confidence-weighted voting. It is an ensemble, not another GNN architecture.
- **Baselines:** Chaitin-Briggs is a classical graph-coloring allocator;
  Random is a simple comparison baseline.
- **Dashboard:** Explore graph allocations, agent insights, agreement, TAC IR,
  and control-flow blocks in a local browser.

## Quick start

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

To train the specialist agents with the quick preset:

```bash
python main.py --mode train-agents --quick
```

To export a fresh multi-agent dataset and serve the dashboard:

```bash
python main.py --mode export-web-data
python -m http.server 8000 --directory web
```

Then open [http://localhost:8000](http://localhost:8000).

## Repository map

| Path | Contents |
| --- | --- |
| `compiler/` | TAC IR, control-flow graphs, liveness, interference graphs, and the Chaitin-Briggs allocator |
| `agents/` | Specialist pattern agents, checkpoint registry, and consensus arbiter |
| `models/` | GNN architectures, shared allocator components, training, and conflict repair |
| `dataset/` | Synthetic compiler program and graph dataset generation |
| `evaluation/` | Benchmarks, metrics, architecture comparisons, and ablation studies |
| `tests/` | Python test suite |
| `web/` | Interactive dashboard and its exported JSON datasets |
| `gnn-regalloc/` | Separate Java compiler and Python ML workflow |
| `report/`, `docs/`, `results/` | Project reports, baseline notes, and generated result artifacts |

## Important distinction

The project contains two related but distinct model groupings:

1. **Specialist agents in the multi-agent workflow:** R-GCN, GAT, GraphSAGE,
   and GCN. Each agent contributes a prediction to the Consensus Arbiter.
2. **Relational architectures in model training/comparison:** R-GCN, R-GAT,
   R-SAGE, and R-GIN. These are selectable neural-network architectures.

The names overlap, but these lists describe different workflows. Chaitin-Briggs
and Random are baselines, while Consensus is an ensemble decision-maker.

## License and project status

This repository is an educational research prototype. Its primary generator
creates synthetic compiler IR; results should not be interpreted as a
production compiler allocator or as a guarantee of performance on real-world
programs. Refer to [PROJECT_GUIDE.md](PROJECT_GUIDE.md) and the reports in
[`report/`](report/) for implementation details and recorded limitations.
