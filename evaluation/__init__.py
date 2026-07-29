"""
Evaluation Package for GNN Register Allocator Benchmarking.
Contains metric computation and comparative benchmark evaluator.
"""

from .metrics import CompilerAllocationMetrics, EvaluatedMetrics
from .evaluator import BenchmarkEvaluator

__all__ = ["CompilerAllocationMetrics", "EvaluatedMetrics", "BenchmarkEvaluator"]
