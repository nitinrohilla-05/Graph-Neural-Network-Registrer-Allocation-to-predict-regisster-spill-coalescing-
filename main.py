"""
Main Entrypoint CLI for Graph Neural Network Register Allocation & Spill Coalescing.
Supports training, evaluation, compiler execution, web visualizer export, JSON dataset generation,
multi-seed model architecture comparison, and feature ablation studies.
"""

import argparse
import json
import sys
import os
import torch
import numpy as np

from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator
from dataset.generator import SyntheticIRGenerator
from dataset.dataset import InterferenceGraphDataset, RegAllocDataset, split_raw_samples
from models.gnn_allocator import RelationalGNNRegisterAllocator
from models.trainer import GNNTrainer
from evaluation.evaluator import BenchmarkEvaluator
from evaluation.ablation import FeatureAblationStudy


def train_mode(args):
    print("=" * 70)
    print(" GNN REGISTER ALLOCATION & SPILL COALESCING - TRAINING PIPELINE ")
    print("=" * 70)

    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed)

    print(f"\n1. Generating synthetic compiler IR dataset ({args.samples} programs, K={num_registers} registers)...")
    raw_dataset = generator.generate_dataset(num_samples=args.samples, num_registers=num_registers)
    raw_train, raw_val, raw_test = split_raw_samples(raw_dataset, seed=args.seed)
    dataset = InterferenceGraphDataset(raw_train, num_registers=num_registers)
    val_dataset = InterferenceGraphDataset(
        raw_val, num_registers=num_registers, normalization_stats=dataset.normalization_stats
    )
    test_dataset = InterferenceGraphDataset(
        raw_test, num_registers=num_registers, normalization_stats=dataset.normalization_stats
    )
    print(f"   Split: train={len(dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    print("\n2. Initializing Relational Graph Neural Network (R-GCN)...")
    model = RelationalGNNRegisterAllocator(
        in_node_features=6,
        hidden_dim=args.hidden_dim,
        num_registers=num_registers,
        num_layers=3
    )

    trainer = GNNTrainer(model, dataset, val_dataset=val_dataset, lr=1e-3, device="cpu")

    print("\n3. Starting Training Loop...")
    trainer.train(num_epochs=args.epochs, verbose=True)
    test_metrics = trainer.evaluate(test_dataset)
    print(f"   Held-out Test Acc: {test_metrics['accuracy']*100:.1f}% | "
          f"Color Violations: {test_metrics['violation_rate_pct']:.2f}%")

    print(f"\n4. Saving Trained Model Checkpoint to '{args.output}'...")
    trainer.save_checkpoint(args.output)

    print("\nTraining Complete! Run '--mode evaluate' to benchmark against baseline allocators.")


def evaluate_mode(args):
    print("=" * 70)
    print(" COMPARATIVE BENCHMARK: GNN vs CHAITIN-BRIGGS vs RANDOM ALLOCATOR ")
    print("=" * 70)

    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed + 100)

    print(f"\nGenerating 30 test programs (K={num_registers} registers)...")
    test_samples = generator.generate_dataset(num_samples=30, num_registers=num_registers)

    if os.path.exists(args.model_path):
        print(f"Loading pre-trained GNN model from '{args.model_path}'...")
        model = GNNTrainer.load_checkpoint(args.model_path, device="cpu")
    else:
        print(f"Checkpoint '{args.model_path}' not found. Initializing untrained GNN for comparison...")
        model = RelationalGNNRegisterAllocator(num_registers=num_registers)

    evaluator = BenchmarkEvaluator(num_registers=num_registers)
    results = evaluator.benchmark_batch(model, test_samples)
    summary = evaluator.summarize_benchmark(results)

    print("\n" + "-" * 65)
    print(f"{'Strategy':<20} | {'Avg Spills':<12} | {'Avg Spill Cost':<15} | {'Move Elim %':<12}")
    print("-" * 65)
    for strategy, metrics in summary.items():
        print(f"{strategy:<20} | {metrics['avg_spills']:<12} | {metrics['avg_spill_cost']:<15} | {metrics['avg_move_elim_pct']:.1f}%")
    print("-" * 65)


def run_compiler_mode(args):
    print("=" * 70)
    print(" COMPILER RUNTIME: GNN-GUIDED REGISTER ALLOCATION & COALESCING ")
    print("=" * 70)

    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed)
    prog = generator.generate_program(num_vars=12, num_instructions=25)

    print(f"\n[1] Program TAC Instructions ({len(prog.instructions)} lines):")
    print("-" * 50)
    print(prog)

    cfg = ControlFlowGraph(prog)
    liveness = LivenessAnalyzer(cfg)
    ig = InterferenceGraph(prog, cfg, liveness)

    print(f"\n[2] Interference & Coalescing Graph Summary:")
    print(f"    - Virtual Registers: {len(ig.variables)}")
    print(f"    - Interference Edges: {len(ig.interference_edges)}")
    print(f"    - Move Coalescing Edges: {len(ig.coalescing_edges)}")

    if os.path.exists(args.model_path):
        print(f"\n[3] Running GNN Allocation (Model loaded from '{args.model_path}')...")
        model = GNNTrainer.load_checkpoint(args.model_path)
    else:
        print(f"\n[3] Running GNN Allocation (Fresh model initialization)...")
        model = RelationalGNNRegisterAllocator(num_registers=num_registers)

    evaluator = BenchmarkEvaluator(num_registers=num_registers)
    gnn_assignment, conflicts, _ = evaluator.run_gnn_allocation(model, ig, apply_repair=True)

    print(f"\n[4] GNN Physical Register Assignments & Spill Decisions:")
    print("-" * 50)
    for var_name, reg in sorted(gnn_assignment.items()):
        status = "STACK SPILL" if reg == "SPILL" else f"Physical Register [{reg}]"
        print(f"    {var_name:<10} -> {status}")

    cb_allocator = ChaitinBriggsAllocator(num_registers=num_registers)
    cb_res = cb_allocator.allocate(ig)

    print(f"\n[5] Allocation Comparison:")
    print(f"    - GNN Spills: {sum(1 for r in gnn_assignment.values() if r == 'SPILL')}")
    print(f"    - Chaitin-Briggs Spills: {len(cb_res.spilled_vars)}")
    print(f"    - Conflicts Repaired Post-Hoc: {conflicts}")


def export_json_mode(args):
    print("=" * 70)
    print(" DATASET EXPORT: RAW JSON INTERFERENCE GRAPHS ")
    print("=" * 70)

    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed)
    output_dir = args.json_dir

    print(f"Generating {args.samples} programs and exporting to '{output_dir}'...")
    filepaths = generator.export_synthetic_dataset_to_json(
        num_samples=args.samples, output_dir=output_dir, num_registers=num_registers
    )
    print(f"Successfully exported {len(filepaths)} JSON graph files to '{output_dir}'!")


def model_comparison_mode(args):
    print("=" * 70)
    print(" PHASE 6 MODEL ARCHITECTURE COMPARISON (MULTI-SEED MEAN +/- STD) ")
    print("=" * 70)

    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed)
    print(f"Generating synthetic dataset ({args.samples} samples, K={num_registers})...")
    raw_dataset = generator.generate_dataset(num_samples=args.samples, num_registers=num_registers)
    raw_train, raw_val, raw_test = split_raw_samples(raw_dataset, seed=args.seed)
    dataset = InterferenceGraphDataset(raw_train, num_registers=num_registers)
    val_dataset = InterferenceGraphDataset(
        raw_val, num_registers=num_registers, normalization_stats=dataset.normalization_stats
    )
    test_samples = raw_test
    print(f"Split: train={len(dataset)}, val={len(val_dataset)}, test={len(test_samples)}")

    evaluator = BenchmarkEvaluator(num_registers=num_registers)
    print("\nRunning multi-seed comparison across GCN, GraphSAGE, GAT, and R-GCN...")
    comp_results = evaluator.compare_model_architectures(
        dataset, val_dataset, test_samples, seeds=[0, 1, 2], epochs=args.epochs
    )

    print("\n" + "=" * 75)
    print(f"{'Architecture':<15} | {'Val Accuracy (%)':<22} | {'Avg Spills':<15} | {'Inf Time (ms)':<15}")
    print("=" * 75)
    for arch, m in comp_results.items():
        acc_str = f"{m['mean_acc']*100:.1f}% +/- {m['std_acc']*100:.1f}%"
        spill_str = f"{m['mean_spills']:.1f} +/- {m['std_spills']:.1f}"
        time_str = f"{m['mean_inference_time_ms']:.2f} ms"
        print(f"{arch:<15} | {acc_str:<22} | {spill_str:<15} | {time_str:<15}")
    print("=" * 75)


def ablation_mode(args):
    print("=" * 70)
    print(" FEATURE IMPORTANCE ABLATION STUDY ")
    print("=" * 70)

    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed)
    print(f"Generating synthetic dataset ({args.samples} samples, K={num_registers})...")
    raw_dataset = generator.generate_dataset(num_samples=args.samples, num_registers=num_registers)
    raw_train, raw_val, _ = split_raw_samples(raw_dataset, seed=args.seed)
    dataset = InterferenceGraphDataset(raw_train, num_registers=num_registers)
    val_dataset = InterferenceGraphDataset(
        raw_val, num_registers=num_registers, normalization_stats=dataset.normalization_stats
    )
    print(f"Split: train={len(dataset)}, val={len(val_dataset)}")

    ablation = FeatureAblationStudy(num_registers=num_registers)
    ablation_results = ablation.run_ablation_study(dataset, val_dataset, epochs=args.epochs, seed=args.seed)

    print("\n" + "-" * 60)
    print(f"{'Feature Subset':<25} | {'Accuracy (%)':<15} | {'Violations (%)':<15}")
    print("-" * 60)
    for subset, metrics in ablation_results.items():
        acc = metrics['accuracy'] * 100.0
        viol = metrics['violation_rate_pct']
        print(f"{subset:<25} | {acc:<15.1f}% | {viol:<15.2f}%")
    print("-" * 60)


def export_web_data(args):
    """Exports sample compiler graph and GNN output to JSON for the Web Visualizer dashboard."""
    num_registers = args.registers
    generator = SyntheticIRGenerator(seed=args.seed)
    prog = generator.generate_program(num_vars=14, num_instructions=28)

    cfg = ControlFlowGraph(prog)
    liveness = LivenessAnalyzer(cfg)
    ig = InterferenceGraph(prog, cfg, liveness)

    if os.path.exists(args.model_path):
        model = GNNTrainer.load_checkpoint(args.model_path)
    else:
        model = RelationalGNNRegisterAllocator(num_registers=num_registers)

    evaluator = BenchmarkEvaluator(num_registers=num_registers)
    gnn_assignment, _, _ = evaluator.run_gnn_allocation(model, ig, apply_repair=True)

    cb_allocator = ChaitinBriggsAllocator(num_registers=num_registers)
    cb_res = cb_allocator.allocate(ig)

    nodes = []
    features = ig.get_node_features()
    for v in ig.variables:
        name = v.name
        feat = features[name]
        nodes.append({
            "id": name,
            "spill_cost": feat.spill_cost,
            "loop_depth": feat.loop_depth,
            "degree": feat.degree,
            "move_degree": feat.move_degree,
            "gnn_assignment": gnn_assignment.get(name, "SPILL"),
            "cb_assignment": cb_res.register_assignment.get(name, "SPILL")
        })

    interf_edges = [{"source": u, "target": v, "type": "interference"} for u, v in ig.interference_edges]
    coal_edges = [{"source": u, "target": v, "type": "coalescing"} for u, v in ig.coalescing_edges]

    instructions = [str(inst) for inst in prog.instructions]

    cfg_blocks = []
    for b in cfg.blocks:
        cfg_blocks.append({
            "block_id": b.block_id,
            "label": b.label,
            "loop_depth": b.loop_depth,
            "instructions": [str(i) for i in b.instructions],
            "successors": [s.block_id for s in b.successors]
        })

    export_payload = {
        "program_name": prog.name,
        "num_registers": num_registers,
        "instructions": instructions,
        "cfg_blocks": cfg_blocks,
        "nodes": nodes,
        "interference_edges": interf_edges,
        "coalescing_edges": coal_edges,
        "cb_coalesced_pairs": cb_res.coalesced_pairs,
        "cb_spills": list(cb_res.spilled_vars),
        "gnn_spills": [name for name, r in gnn_assignment.items() if r == "SPILL"]
    }

    out_file = args.web_output
    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"Successfully exported web visualizer dataset to '{out_file}'!")


def main():
    parser = argparse.ArgumentParser(
        description="Graph Neural Network Register Allocation & Spill Coalescing Platform"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "run-compiler", "export-web-data", "export-json", "model-comparison", "ablation"],
        default="train",
        help="Pipeline execution mode"
    )
    parser.add_argument("--registers", type=int, default=4, help="Number of physical registers K")
    parser.add_argument("--samples", type=int, default=60, help="Number of dataset samples for training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epoch count")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="gnn_allocator.pt", help="Model checkpoint path")
    parser.add_argument("--model-path", type=str, default="gnn_allocator.pt", help="Model checkpoint path to load")
    parser.add_argument("--json-dir", type=str, default="data/raw_graphs", help="Directory for JSON graph files")
    parser.add_argument("--web-output", type=str, default="web/data.json", help="Export path for web dashboard data")

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "run-compiler":
        run_compiler_mode(args)
    elif args.mode == "export-web-data":
        export_web_data(args)
    elif args.mode == "export-json":
        export_json_mode(args)
    elif args.mode == "model-comparison":
        model_comparison_mode(args)
    elif args.mode == "ablation":
        ablation_mode(args)


if __name__ == "__main__":
    main()
