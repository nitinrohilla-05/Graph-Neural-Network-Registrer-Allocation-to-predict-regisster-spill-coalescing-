"""
Dataset package for GNN Register Allocator.
Generates synthetic TAC programs and creates PyTorch Graph datasets.
"""

from .generator import SyntheticIRGenerator
from .dataset import InterferenceGraphDataset, GraphDataSample

__all__ = ["SyntheticIRGenerator", "InterferenceGraphDataset", "GraphDataSample"]
