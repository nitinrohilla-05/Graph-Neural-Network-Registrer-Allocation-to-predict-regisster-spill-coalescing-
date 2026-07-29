"""
Comparative Benchmark Evaluator for Register Allocation Strategies.
Compares GNN-guided Allocators (GCN, GraphSAGE, GAT, R-GCN) vs Chaitin-Briggs Baseline vs Random Allocator.
Tracks pre-repair vs post-repair validity, conflict rates, wall-clock timing, and multi-seed statistics.
"""

from typing import List, Dict, Tuple, Any
import random
import time
import torch
import torch.nn as nn
import numpy as np

from compiler.ir import Program
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator
from dataset.generator import SyntheticIRGenerator
from models.gnn_allocator import (
    RelationalGNNRegisterAllocator,
    GCNSpillPredictor,
    SAGESpillPredictor,
    GATSpillPredictor
)
from models.repair import repair_conflicts
from models.trainer import GNNTrainer
from evaluation.metrics import CompilerAllocationMetrics, EvaluatedMetrics


class BenchmarkEvaluator:
    """Runs end-to-end benchmark comparisons across multiple programs."""
    def __init__(self, num_registers: int = 4):
        self.num_registers: int = num_registers
        self.registers: List[str] = [f"R{i}" for i in range(num_registers)]

    def run_gnn_allocation(
        self, model: nn.Module, ig: InterferenceGraph, apply_repair: bool = True
    ) -> Tuple[Dict[str, str], int, float]:
        """
        Infers register assignments and spill decisions for an interference graph using GNN.
        Returns: (assignment dict, pre_repair_conflicts_found, inference_time_seconds)
        """
        model.eval()
        t0 = time.perf_counter()

        feat_matrix = ig.get_feature_matrix()
        x = torch.tensor(feat_matrix, dtype=torch.float32)

        N = len(ig.variables)
        var_names = [v.name for v in ig.variables]
        var_to_idx = {v.name: i for i, v in enumerate(ig.variables)}

        interf_adj = torch.zeros((N, N), dtype=torch.float32)
        for u, v in ig.interference_edges:
            if u in var_to_idx and v in var_to_idx:
                i, j = var_to_idx[u], var_to_idx[v]
                interf_adj[i, j] = 1.0
                interf_adj[j, i] = 1.0

        coal_adj = torch.zeros((N, N), dtype=torch.float32)
        for u, v in ig.coalescing_edges:
            if u in var_to_idx and v in var_to_idx:
                i, j = var_to_idx[u], var_to_idx[v]
                coal_adj[i, j] = 1.0
                coal_adj[j, i] = 1.0

        with torch.no_grad():
            color_logits, _ = model(x, interf_adj, coal_adj)
            pred_classes = torch.argmax(color_logits, dim=-1)  # [N]

        # Greedy Conflict Repair (Phase 5)
        if apply_repair:
            repaired_classes, conflicts_found = repair_conflicts(
                interf_adj, pred_classes, num_registers=self.num_registers, spill_class=self.num_registers
            )
        else:
            repaired_classes = pred_classes
            conflicts_found = 0

        t1 = time.perf_counter()
        inference_time = t1 - t0

        assignment: Dict[str, str] = {}
        for name in var_names:
            idx = var_to_idx[name]
            cls_idx = repaired_classes[idx].item()
            if cls_idx < self.num_registers:
                assignment[name] = f"R{cls_idx}"
            else:
                assignment[name] = "SPILL"

        return assignment, conflicts_found, inference_time

    def run_random_allocation(self, ig: InterferenceGraph) -> Dict[str, str]:
        """Random baseline allocator."""
        g_interf, _ = ig.to_networkx()
        assignment: Dict[str, str] = {}

        for v in ig.variables:
            neighbor_colors = {
                assignment[nbr] for nbr in g_interf.neighbors(v.name) if nbr in assignment
            }
            avail = [r for r in self.registers if r not in neighbor_colors]
            if avail and random.random() > 0.3:
                assignment[v.name] = random.choice(avail)
            else:
                assignment[v.name] = "SPILL"

        return assignment

    def benchmark_batch(
        self,
        model: nn.Module,
        test_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, Any]]
    ) -> Dict[str, List[EvaluatedMetrics]]:
        """Runs comparative benchmark across test dataset samples."""
        results: Dict[str, List[EvaluatedMetrics]] = {
            "GNN": [],
            "Chaitin-Briggs": [],
            "Random": []
        }

        cb_allocator = ChaitinBriggsAllocator(num_registers=self.num_registers)

        for prog, cfg, liveness, ig, _ in test_samples:
            # 1. Chaitin-Briggs Baseline
            cb_result = cb_allocator.allocate(ig)
            cb_metrics = CompilerAllocationMetrics.evaluate(
                "Chaitin-Briggs", ig, cb_result.register_assignment, cb_result.coalesced_pairs, self.num_registers
            )
            results["Chaitin-Briggs"].append(cb_metrics)

            # 2. GNN Allocator
            gnn_assignment, _, _ = self.run_gnn_allocation(model, ig, apply_repair=True)
            gnn_metrics = CompilerAllocationMetrics.evaluate(
                "GNN-Guided", ig, gnn_assignment, [], self.num_registers
            )
            results["GNN"].append(gnn_metrics)

            # 3. Random Baseline
            rand_assignment = self.run_random_allocation(ig)
            rand_metrics = CompilerAllocationMetrics.evaluate(
                "Random", ig, rand_assignment, [], self.num_registers
            )
            results["Random"].append(rand_metrics)

        return results

    def compare_model_architectures(
        self,
        dataset: Any,
        test_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, Any]],
        seeds: List[int] = [0, 1, 2],
        epochs: int = 15
    ) -> Dict[str, Dict[str, float]]:
        """
        Phase 6 Model Comparison:
        Evaluates GCN, GraphSAGE, GAT, and R-GCN over multiple random seeds, reporting mean +- std.
        """
        architectures = [
            ("GCN", GCNSpillPredictor),
            ("GraphSAGE", SAGESpillPredictor),
            ("GAT", GATSpillPredictor),
            ("R-GCN", RelationalGNNRegisterAllocator)
        ]

        model_results = {}

        for name, cls in architectures:
            f1_list = []
            acc_list = []
            spill_list = []
            time_list = []

            for s in seeds:
                torch.manual_seed(s)
                model = cls(in_channels=6, num_registers=self.num_registers)
                trainer = GNNTrainer(model, dataset)
                trainer.train(num_epochs=epochs, verbose=False)

                eval_metrics = trainer.evaluate()
                acc_list.append(eval_metrics["accuracy"])

                # Run timing and spill benchmark on test samples
                spill_count = 0
                t_total = 0.0
                for prog, cfg, liveness, ig, _ in test_samples:
                    asgn, _, t_infer = self.run_gnn_allocation(model, ig, apply_repair=True)
                    spill_count += sum(1 for r in asgn.values() if r == "SPILL")
                    t_total += t_infer

                spill_list.append(spill_count / max(len(test_samples), 1))
                time_list.append(t_total / max(len(test_samples), 1))

            model_results[name] = {
                "mean_acc": float(np.mean(acc_list)),
                "std_acc": float(np.std(acc_list)),
                "mean_spills": float(np.mean(spill_list)),
                "std_spills": float(np.std(spill_list)),
                "mean_inference_time_ms": float(np.mean(time_list) * 1000.0),
            }

        return model_results

    @staticmethod
    def summarize_benchmark(results: Dict[str, List[EvaluatedMetrics]]) -> Dict[str, Dict[str, float]]:
        """Computes aggregate benchmark averages across test programs."""
        summary = {}
        for strategy, metrics_list in results.items():
            n = len(metrics_list)
            avg_spills = sum(m.total_spills for m in metrics_list) / max(n, 1)
            avg_spill_cost = sum(m.total_spill_cost for m in metrics_list) / max(n, 1)
            avg_move_elim = sum(m.move_elimination_rate_pct for m in metrics_list) / max(n, 1)
            avg_conflicts = sum(m.coloring_conflicts for m in metrics_list) / max(n, 1)

            summary[strategy] = {
                "avg_spills": round(avg_spills, 2),
                "avg_spill_cost": round(avg_spill_cost, 2),
                "avg_move_elim_pct": round(avg_move_elim, 2),
                "avg_conflicts": round(avg_conflicts, 2)
            }
        return summary
