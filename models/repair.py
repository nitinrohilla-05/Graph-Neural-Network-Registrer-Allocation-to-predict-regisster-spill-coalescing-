"""
Post-Hoc Greedy Conflict Resolution Module for GNN Register Allocation.
Implements greedy post-processing to eliminate coloring conflicts on raw GNN predictions.
"""

from typing import Tuple, Dict, Set, List
import torch


def repair_conflicts(
    edge_index_or_adj: torch.Tensor,
    predicted_labels: torch.Tensor,
    num_registers: int = 4,
    spill_class: int = 4
) -> Tuple[torch.Tensor, int]:
    """
    Greedy repair post-processing (Phase 5):
    For each interference edge with matching non-spill labels, reassigns one endpoint to the
    lowest available non-conflicting physical register, or spills it if none is available.

    Args:
        edge_index_or_adj: [2, E] edge index tensor OR [N, N] adjacency matrix tensor.
        predicted_labels: [N] predicted register/spill class indices.
        num_registers: number of physical registers K.
        spill_class: index of spill class (usually K).

    Returns:
        labels: [N] repaired register/spill class tensor.
        conflicts_found: total number of coloring conflicts identified and repaired.
    """
    labels = predicted_labels.clone()
    N = len(labels)

    # Build adjacency list
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(N)}
    if edge_index_or_adj.dim() == 2 and edge_index_or_adj.size(0) == 2:
        # Edge index [2, E]
        edges = edge_index_or_adj.t().tolist()
        for u, v in edges:
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)
    elif edge_index_or_adj.dim() == 2 and edge_index_or_adj.size(0) == edge_index_or_adj.size(1):
        # Adjacency matrix [N, N]
        adj = edge_index_or_adj.nonzero(as_tuple=False).tolist()
        for u, v in adj:
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)

    conflicts_found = 0
    # Process nodes in order of degree (highest degree first)
    order = sorted(range(N), key=lambda n: -len(adjacency[n]))

    for node in order:
        current_color = labels[node].item()
        if current_color == spill_class:
            continue

        neighbor_labels = {labels[nbr].item() for nbr in adjacency[node]}
        if current_color in neighbor_labels:
            conflicts_found += 1
            available = [r for r in range(num_registers) if r not in neighbor_labels]
            if available:
                labels[node] = available[0]
            else:
                labels[node] = spill_class

    return labels, conflicts_found
