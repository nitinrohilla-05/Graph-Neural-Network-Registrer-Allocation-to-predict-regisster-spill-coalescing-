"""
GNN Model Package for Register Allocation & Spill Coalescing.
Contains Relational GNN Architecture and Training Pipeline.
"""

from .gnn_allocator import RelationalGNNRegisterAllocator, GraphColoringLoss
from .trainer import GNNTrainer

__all__ = ["RelationalGNNRegisterAllocator", "GraphColoringLoss", "GNNTrainer"]
