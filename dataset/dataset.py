"""
PyTorch Dataset module for Compiler Interference Graphs.
Converts TAC IR graphs, node features, adjacency matrices, and ground truth labels into PyTorch tensors.
Supports both in-memory compiler objects and raw JSON file datasets (Phase 2.5 schema).
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from compiler.ir import Program
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import AllocationResult


class GraphDataSample:
    """Dataclass storing PyTorch Tensor representations for a single interference graph."""
    def __init__(
        self,
        node_features: torch.Tensor,       # [N, F] - float32
        interf_adj: torch.Tensor,          # [N, N] - float32 adjacency matrix
        coal_adj: torch.Tensor,            # [N, N] - float32 coalescing adjacency matrix
        target_colors: torch.Tensor,       # [N] - int64 class labels (0..K-1 = registers, K = spill)
        coalesce_labels: torch.Tensor,     # [N, N] - float32 edge coalescing targets (1 if pair coalesced)
        var_names: List[str],              # List of variable names
        num_registers: int                 # Physical register count K
    ):
        self.node_features: torch.Tensor = node_features
        self.interf_adj: torch.Tensor = interf_adj
        self.coal_adj: torch.Tensor = coal_adj
        self.target_colors: torch.Tensor = target_colors
        self.coalesce_labels: torch.Tensor = coalesce_labels
        self.var_names: List[str] = var_names
        self.num_registers: int = num_registers

    @property
    def y_spill(self) -> torch.Tensor:
        """Returns binary spill tensor [N] (1.0 if spill, 0.0 if assigned register)."""
        return (self.target_colors == self.num_registers).float()


class InterferenceGraphDataset(Dataset):
    """PyTorch Dataset wrapping generated compiler interference graphs."""
    def __init__(
        self,
        raw_samples: List[Tuple[Program, ControlFlowGraph, LivenessAnalyzer, InterferenceGraph, AllocationResult]],
        num_registers: int = 4
    ):
        self.num_registers: int = num_registers
        self.samples: List[GraphDataSample] = []

        for prog, cfg, liveness, ig, gt in raw_samples:
            sample = self._convert_sample(ig, gt, num_registers)
            self.samples.append(sample)

    def _convert_sample(self, ig: InterferenceGraph, gt: AllocationResult, num_registers: int) -> GraphDataSample:
        N = len(ig.variables)
        var_names = [v.name for v in ig.variables]
        var_to_idx = {v.name: i for i, v in enumerate(ig.variables)}

        feat_matrix = ig.get_feature_matrix()
        x = torch.tensor(feat_matrix, dtype=torch.float32)

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

        target_colors = torch.zeros(N, dtype=torch.long)
        for name, idx in var_to_idx.items():
            if name in gt.register_assignment:
                reg_str = gt.register_assignment[name]  # e.g., 'R0', 'R1'
                reg_num = int(reg_str.replace("R", ""))
                target_colors[idx] = min(reg_num, num_registers - 1)
            else:
                target_colors[idx] = num_registers  # Class K is Spill

        coalesce_labels = torch.zeros((N, N), dtype=torch.float32)
        for u, v in gt.coalesced_pairs:
            if u in var_to_idx and v in var_to_idx:
                i, j = var_to_idx[u], var_to_idx[v]
                coalesce_labels[i, j] = 1.0
                coalesce_labels[j, i] = 1.0

        return GraphDataSample(
            node_features=x,
            interf_adj=interf_adj,
            coal_adj=coal_adj,
            target_colors=target_colors,
            coalesce_labels=coalesce_labels,
            var_names=var_names,
            num_registers=num_registers
        )

    def get_pos_weight(self) -> torch.Tensor:
        """Computes pos_weight = (num_negative / num_positive) to handle spill class imbalance."""
        total_spills = 0
        total_non_spills = 0
        for sample in self.samples:
            spills = torch.sum(sample.y_spill).item()
            total_spills += spills
            total_non_spills += (len(sample.target_colors) - spills)
        pos_weight = float(total_non_spills) / max(float(total_spills), 1.0)
        return torch.tensor([pos_weight], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphDataSample:
        return self.samples[idx]


class RegAllocDataset(Dataset):
    """
    JSON File Dataset Loader conforming to Phase 4.1 PyG specification.
    Loads raw JSON interference graphs from directory.
    """
    def __init__(self, json_dir: str):
        super().__init__()
        self.json_dir = json_dir
        self.files = sorted(list(Path(json_dir).glob("*.json")))
        self.samples: List[GraphDataSample] = []
        self._load_files()

    def _load_files(self):
        for fpath in self.files:
            with open(fpath, "r") as f:
                g = json.load(f)

            num_registers = g.get("num_registers", 4)
            nodes = g["nodes"]
            N = len(nodes)

            # Node features [N, 6]: [spill_cost, loop_depth, degree, move_degree, live_range_length, use_def_count]
            raw_feats = []
            for n in nodes:
                is_move = 1.0 if n.get("is_move_related", False) else 0.0
                raw_feats.append([
                    n.get("spill_cost", 1.0),
                    float(n.get("loop_depth", 0)),
                    float(n.get("degree", 0)),
                    is_move,
                    float(n.get("live_range_length", 1)),
                    float(n.get("use_def_count", 1))
                ])

            feat_matrix = np.array(raw_feats, dtype=np.float32)
            stds = np.std(feat_matrix, axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            means = np.mean(feat_matrix, axis=0, keepdims=True)
            norm_feats = (feat_matrix - means) / stds
            x = torch.tensor(norm_feats, dtype=torch.float32)

            interf_adj = torch.zeros((N, N), dtype=torch.float32)
            for u, v in g.get("interference_edges", []):
                if u < N and v < N:
                    interf_adj[u, v] = 1.0
                    interf_adj[v, u] = 1.0

            coal_adj = torch.zeros((N, N), dtype=torch.float32)
            coalesce_labels = torch.zeros((N, N), dtype=torch.float32)
            for me in g.get("move_edges", []):
                u, v = me["pair"]
                if u < N and v < N:
                    coal_adj[u, v] = 1.0
                    coal_adj[v, u] = 1.0
                    if me.get("coalesce_safe", False):
                        coalesce_labels[u, v] = 1.0
                        coalesce_labels[v, u] = 1.0

            target_colors = torch.zeros(N, dtype=torch.long)
            var_names = []
            for i, n in enumerate(nodes):
                var_names.append(n.get("name", f"v{i}"))
                if n.get("label_spill", False) or n.get("label_register") is None:
                    target_colors[i] = num_registers
                else:
                    target_colors[i] = min(int(n["label_register"]), num_registers - 1)

            sample = GraphDataSample(
                node_features=x,
                interf_adj=interf_adj,
                coal_adj=coal_adj,
                target_colors=target_colors,
                coalesce_labels=coalesce_labels,
                var_names=var_names,
                num_registers=num_registers
            )
            self.samples.append(sample)

    def get_pos_weight(self) -> torch.Tensor:
        total_spills = 0
        total_non_spills = 0
        for sample in self.samples:
            spills = torch.sum(sample.y_spill).item()
            total_spills += spills
            total_non_spills += (len(sample.target_colors) - spills)
        pos_weight = float(total_non_spills) / max(float(total_spills), 1.0)
        return torch.tensor([pos_weight], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphDataSample:
        return self.samples[idx]
