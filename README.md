# Multi-Agent Graph Neural Network Register Allocation & Spill Coalescing Platform

> **College Capstone Project: Compiler Design + Machine Learning for Compilers (ML4Compilers)**  
> An end-to-end multi-agent compiler optimization platform where **4 distinct GNN pattern-detection agents** collaborate through a central **Consensus Arbiter** to solve NP-complete Register Allocation, Spill Prediction, and Move Coalescing, with side-by-side benchmarking across **4 multi-relational GNN architectures** (R-GCN, R-GAT, R-SAGE, R-GIN).

---

## 📌 Project Overview

Register Allocation is a critical code-generation phase in modern production compilers (LLVM, GCC). It maps an unbound set of **Virtual Registers** (live ranges) from Intermediate Representation (IR) onto a fixed set of $K$ **Physical Processor Registers** (e.g., $K = 4, 8, 16$).

When physical registers are insufficient:
1. **Spilling**: Variables must be stored in the stack frame, incurring memory traffic and cache misses.
2. **Coalescing**: Variables linked by `MOVE` instructions ($v_{\text{dst}} \leftarrow v_{\text{src}}$) should share the same register to eliminate redundant copy instructions.

Traditional compilers rely on greedy heuristics like the **Chaitin-Briggs Graph Coloring** algorithm. In this project, we partition compiler pattern detection across **4 specialized AI Agents** (one per team member) combined via a **Consensus Arbiter**, achieving reliable, conflict-free register allocation with an interactive visualizer dashboard and rigorous multi-model comparison.

---

## 👥 Team & Agent Ownership Structure

To fulfill course requirements for a 4-person engineering team, pattern recognition is divided among 4 specialized agents:

| Teammate | Agent Class | Model Architecture | Specialized Compiler Pattern Detected |
| :--- | :--- | :--- | :--- |
| **Member 1** (`[Teammate 1 Name]`) | `RelationalAgent` | **R-GCN** (Relational Graph Convolution) | Multi-relational move-chains, affinity copy edges, and coalescing opportunities. |
| **Member 2** (`[Teammate 2 Name]`) | `AttentionAgent` | **GAT** (Graph Attention Network) | Directed neighbor pressure, top-attended interference edges pushing nodes to spill. |
| **Member 3** (`[Teammate 3 Name]`) | `NeighbourhoodAgent` | **GraphSAGE** (Mean Aggregator) | Local neighborhood density, high-degree hub nodes, and dense $K$-cliques. |
| **Member 4** (`[Teammate 4 Name]`) | `PressureAgent` | **GCN** (Symmetric Laplacian) | Global register pressure peaks across CFG basic blocks and loop-hot variables. |
| **Ensemble** | `ConsensusArbiter` | **Two-Stage Soft Voting** | Confidence-weighted ensemble arbitration, Briggs/George safe coalescing, and greedy repair. |

---

## 🏗️ Multi-Agent Architecture

```
                                  +------------------------------------+
                                  |    Three-Address Code (TAC IR)     |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |    CFG & Loop-Depth Analysis       |
                                  |    Backward Dataflow Liveness      |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |    Multi-Relational Graph (IG)     |
                                  |  (Interference + Coalescing Edges) |
                                  +-----------------+------------------+
                                                    |
                    +-------------------------------+-------------------------------+
                    |                               |                               |                               |
                    v                               v                               v                               v
        +-----------------------+       +-----------------------+       +-----------------------+       +-----------------------+
        |  Member 1: RGCN       |       |  Member 2: GAT        |       |  Member 3: GraphSAGE  |       |  Member 4: GCN        |
        |  RelationalAgent      |       |  AttentionAgent       |       |  NeighbourhoodAgent   |       |  PressureAgent        |
        |  (Move-Chains)        |       |  (Neighbour Pressure) |       |  (Hubs & Cliques)     |       |  (Loop-Hot Nodes)     |
        +-----------+-----------+       +-----------+-----------+       +-----------+-----------+       +-----------+-----------+
                    |                               |                               |                               |
                    +-------------------------------+-------------------------------+-------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |         Consensus Arbiter          |
                                  |  1. Entropy Confidence Weights     |
                                  |  2. Soft-Voting Spill vs Keep      |
                                  |  3. Physical Register Selection    |
                                  |  4. Briggs/George Safe Coalescing  |
                                  |  5. 100% Conflict Greedy Repair    |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Valid Register Assignment + Spills |
                                  | Interactive Web Visualizer Dashboard|
                                  +------------------------------------+
```

---

## 🔬 Four Multi-Relational Architectures

In addition to the multi-agent ensemble, the project provides 4 first-class multi-relational GNN models subclassed from `BaseRelationalAllocator` (`models/base_allocator.py`):

1. **R-GCN (`models/gnn_allocator.py`)**: Normalized linear relation-specific message passing over interference and move edges.
2. **R-GAT (`models/rgat_allocator.py`)**: Relation-specific self-attention (`RelationalGATConv`) computing dynamic edge importance weights.
3. **R-SAGE (`models/rsage_allocator.py`)**: Relation-specific mean neighborhood aggregation with self-loop projection (`RelationalSAGEConv`).
4. **R-GIN (`models/gin_allocator.py`)**: Relation-aware sum aggregation with multi-layer perceptron (MLP) update for maximal graph structure expressiveness.

All models share the same differentiable graph coloring loss (`models/base_allocator.py`) combining cross-entropy classification, interference conflict penalty, and move-coalescing bonus.

---

## 📁 Repository Structure

```
.
├── README.md                   # Comprehensive project documentation
├── requirements.txt            # Python dependencies (PyTorch, NetworkX, NumPy, etc.)
├── main.py                     # Unified CLI entrypoint with all operational modes
├── compiler/                   # Core Compiler Subsystem
│   ├── ir.py                   # Three-Address Code (TAC) IR & Instruction definitions
│   ├── cfg.py                  # Control Flow Graph & loop nesting depth analyzer
│   ├── liveness.py             # Backward dataflow liveness analysis
│   ├── interference_graph.py   # Multi-relational interference & coalescing graph builder
│   └── chaitin_briggs.py       # Chaitin-Briggs heuristic baseline & George/Briggs coalescer
├── agents/                     # 4 Pattern Detection Agents & Consensus Module
│   ├── base.py                 # Abstract PatternAgent base class & entropy-confidence utility
│   ├── rgcn_agent.py           # Member 1: RelationalAgent (R-GCN)
│   ├── gat_agent.py            # Member 2: AttentionAgent (GAT)
│   ├── sage_agent.py           # Member 3: NeighbourhoodAgent (GraphSAGE)
│   ├── gcn_agent.py            # Member 4: PressureAgent (GCN)
│   ├── consensus.py            # ConsensusArbiter (confidence-weighted soft-voting ensemble)
│   └── registry.py             # Architecture factory & checkpoint loader
├── dataset/                    # Machine Learning Data Subsystem
│   ├── generator.py            # Synthetic compiler TAC program generator
│   └── dataset.py              # PyTorch Dataset wrapping interference graphs
├── models/                     # Deep Learning Architectures
│   ├── base_allocator.py       # Common BaseRelationalAllocator & shared GraphColoringLoss
│   ├── gnn_allocator.py        # RelationalGNNRegisterAllocator (R-GCN) & baseline models
│   ├── rgat_allocator.py       # RelationalGATRegisterAllocator (R-GAT with relational attention)
│   ├── rsage_allocator.py      # RelationalSAGERegisterAllocator (R-SAGE with mean aggregation)
│   ├── gin_allocator.py        # RelationalGINRegisterAllocator (R-GIN with relational MLP)
│   ├── trainer.py              # GNN training loop & checkpoint serialization (--model support)
│   └── repair.py               # Post-inference greedy conflict repair
├── evaluation/                 # Metrics & Benchmarking Subsystem
│   ├── metrics.py              # Compiler metrics (Spills, Spill Costs, Move Elim %, Conflicts)
│   ├── evaluator.py            # Comparative evaluation & multi-seed model comparisons
│   ├── model_comparator.py     # Cross-architecture pattern comparator & publication plotter
│   └── ablation.py             # Feature importance ablation studies
├── checkpoints/                # Trained model weights (rgcn.pt, rgat.pt, rsage.pt, gin.pt)
├── results/                    # Generated benchmark summaries and reports
├── report/                     # Technical reports & publication figures
│   ├── phase1_background.md    # Initial compiler & GNN background
│   ├── phase2_multi_agent.md   # Multi-agent architecture & multi-model comparison report
│   └── model_comparison.png    # 300 DPI comparative agreement & performance figure
├── web/                        # Interactive Visualizer Dashboard
│   ├── index.html              # Modern glassmorphism web layout & dataset switcher
│   ├── style.css               # Dark-theme styling & responsive design
│   ├── app.js                  # Dynamic canvas engine, force layout, zoom/pan/drag, matrix tables
│   ├── data.json               # Exported multi-agent graph data
│   └── comparison_data.json    # Exported 4-architecture cross-comparison graph data
└── tests/                      # Comprehensive Unit & Integration Test Suite (39 tests)
    ├── test_compiler.py        # IR, CFG, Liveness, and Interference Graph tests
    ├── test_chaitin.py         # Chaitin-Briggs baseline allocator tests
    ├── test_gnn.py             # Pure PyTorch GNN layer & forward pass tests
    ├── test_repair.py          # Greedy conflict repair tests
    ├── test_agents.py          # Pattern detection agent interfaces & serialization tests
    ├── test_consensus.py       # Consensus Arbiter soft-voting & safe coalescing tests
    ├── test_end_to_end.py      # End-to-end dataset & training tests
    └── test_relational_models.py # R-GAT, R-SAGE, R-GIN & ModelComparator tests
```

---

## 🧮 Theoretical & Mathematical Foundations

### 1. Normalized Entropy Confidence Score
Each agent produces softmax probabilities $P(c_v = k)$ across $K+1$ classes (where class $K$ represents **SPILL**). The normalized prediction confidence $C(v) \in [0, 1]$ is defined by:
$$H(v) = - \sum_{k=0}^{K} P(c_v = k) \ln \left( P(c_v = k) + \epsilon \right)$$
$$C(v) = 1 - \frac{H(v)}{\ln(K+1)}$$
A confident, low-entropy prediction yields $C(v) \to 1.0$, whereas maximum uncertainty yields $C(v) \to 0.0$.

### 2. Two-Stage Confidence-Weighted Soft Voting
1. **Stage 1 (Spill Decision)**: Agents vote on whether variable $v$ must spill:
   $$W_{\text{spill}}(v) = \frac{\sum_{a} C_a(v) \cdot P_a(c_v = \text{SPILL})}{\sum_{a} C_a(v)}$$
   If $W_{\text{spill}}(v) > 0.5$, variable $v$ is designated as a stack spill.
2. **Stage 2 (Register Assignment)**: For retained variables, agents soft-vote across physical registers $0 \dots K-1$:
   $$W_{\text{reg}}(v, k) = \frac{\sum_{a} C_a(v) \cdot P_a(c_v = k)}{\sum_{a} C_a(v)}$$
   $$\text{Assign}(v) = \arg\max_{k \in \{0, \dots, K-1\}} W_{\text{reg}}(v, k)$$
   Ties are broken deterministically using variable indices and lowest register ID.

### 3. Conservative Coalescing (Briggs & George Rules)
To guarantee that coalescing two move-related variables $u$ and $v$ does not turn a $K$-colorable graph into an uncolorable one:
- **Briggs Rule**: Coalesce $u$ and $v$ if the merged node has fewer than $K$ neighbors of significant degree ($\ge K$).
- **George Rule**: Coalesce $u$ and $v$ if every neighbor $t$ of $u$ either interferes with $v$ or has degree $< K$.

---

## 🚀 CLI Commands & Usage Guide

### 1. Run Complete Test Suite
Verify that all 39 test cases pass:
```bash
python -m pytest
# or
python -m unittest discover -s tests
```

### 2. Train Any of the 4 Relational GNN Architectures
Train an individual architecture using `--model {rgcn, rgat, rsage, gin}`:
```bash
python main.py --mode train --model rgcn --epochs 20 --samples 60 --registers 4
python main.py --mode train --model rgat --epochs 20 --samples 60 --registers 4
python main.py --mode train --model rsage --epochs 20 --samples 60 --registers 4
python main.py --mode train --model gin --epochs 20 --samples 60 --registers 4
```

### 3. Cross-Architecture Pattern Comparison
Evaluate all 4 trained architectures (plus Chaitin-Briggs and Random) on the exact same compiler graph, computing pairwise register agreement, spill decision confusion matrices, and coalescing overlap:
```bash
# Compare on a fixed test program and generate report/model_comparison.png:
python main.py --mode compare-models --registers 4

# Statistically aggregate over 20 programs:
python main.py --mode compare-models --samples 20 --registers 4
```

### 4. Train All 4 Pattern Detection Agents (Multi-Agent Team Workflow)
Train checkpoints for `rgcn`, `gat`, `sage`, and `gcn`:
```bash
# Quick smoke preset:
python main.py --mode train-agents --quick --checkpoint-dir checkpoints

# Full training run:
python main.py --mode train-agents --samples 60 --epochs 20 --registers 4 --checkpoint-dir checkpoints
```

### 5. Evaluate Agents & Comparative Benchmarks
Compare all 4 agents against the Consensus Arbiter and Chaitin-Briggs baseline:
```bash
python main.py --mode evaluate-agents --quick
```

### 6. Generate Benchmark Reports
Produce `results/agents_results.json` and `results/agents_results.md`:
```bash
python main.py --mode agents-report --quick
```

### 7. Export Data & Launch Interactive Visualizer
Export the multi-agent graph data:
```bash
python main.py --mode export-web-data --registers 4
```
Start the local dashboard server:
```bash
python -m http.server 8000 --directory web
```
Open **[http://localhost:8000](http://localhost:8000)** in any modern web browser. Use the **"Load Comparison Run"** button in the navbar to toggle between the Multi-Agent ensemble view and the 4-Architecture cross-comparison view.

---

## 📊 Benchmark Results

### 1. Multi-Agent Consensus Benchmark ($K=4$, 100% Validity via Greedy Conflict Repair)

| Model / Strategy | Architecture | Specialty Pattern | Avg Spills | Avg Spill Cost | Move Elim (%) | Inference (ms) | Coloring Validity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Relational Agent** | R-GCN | Move-chains & affinity edges | 12.45 | 237.10 | 0.0% | 13.56 ms | 100.0% |
| **Attention Agent** | GAT | Directed neighbor pressure & choking | 15.65 | 377.10 | 0.0% | 11.83 ms | 100.0% |
| **Neighbourhood Agent** | GraphSAGE | Hub nodes & dense K-cliques | 15.65 | 377.10 | 0.0% | 10.96 ms | 100.0% |
| **Pressure Agent** | GCN | Global pressure peaks & loop hotness | 15.65 | 377.10 | 0.0% | 6.95 ms | 100.0% |
| **Consensus Arbiter** | Ensemble | Confidence-weighted soft voting | **11.00** | **229.95** | 0.0% | 6.39 ms | **100.0%** |
| **Chaitin-Briggs** | Classical | Greedy spill cost / degree heuristic | 7.65 | 162.80 | 5.8% | 1.50 ms | 100.0% |

### 2. Cross-Architecture Comparative Benchmark on Identical Input Program

Evaluated on `func_v11_5506` ($N=11$ virtual variables, 33 interference edges, 4 move coalescing edges):

| Model | Architecture | Spills | Spill Cost | Moves Eliminated | Move Elim (%) | Inference (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **R-GCN** | Relational Graph Conv | 6 | 58.0 | 1 | 25.0% | 14.02 ms |
| **R-GAT** | Relational Attention | 11 | 188.0 | 0 | 0.0% | 17.96 ms |
| **R-SAGE** | Relational GraphSAGE | 10 | 145.0 | 0 | 0.0% | 9.71 ms |
| **R-GIN** | Relation-Aware GIN | 11 | 188.0 | 0 | 0.0% | 8.59 ms |
| **Chaitin-Briggs** | Heuristic Baseline | 5 | 48.0 | 0 | 0.0% | 1.17 ms |
| **Random** | Stochastic Baseline | 2 | 22.0 | 0 | 0.0% | 0.05 ms |

**Pairwise Register Agreement Matrix (%)**:
- R-GCN vs R-GAT: 54.5%
- R-GCN vs R-SAGE: 54.5%
- R-GCN vs R-GIN: 54.5%
- R-GAT vs R-GIN: 100.0%
- R-GAT vs R-SAGE: 90.9%

**Publication Figure**: A 300-DPI high-resolution figure combining the pairwise agreement heatmap, spill comparison bar chart, and spill cost chart is saved in [`report/model_comparison.png`](file:///e:/Graph%20Neural%20Network/report/model_comparison.png).

---

## ⚠️ Limitations & Future Work

1. **Synthetic IR**: The dataset generator produces synthetic TAC programs. Real-world LLVM bitcode workloads feature diverse memory access patterns and function call conventions that would provide richer training distributions.
2. **Greedy Heuristic Performance Gap**: Chaitin-Briggs currently incurs fewer spills on dense synthetic graphs because its simplification phase can eliminate low-degree nodes unconditionally. Training on larger datasets with reinforcement learning (PPO) or imitation learning over optimal integer linear programming (ILP) solvers will help bridge this gap.
3. **Target Architectures**: Currently supports generic RISC-like register files with uniform register costs. Future extensions can incorporate calling conventions (caller-saved vs callee-saved registers) and aliased sub-registers (e.g., `RAX`/`EAX`/`AX`).
