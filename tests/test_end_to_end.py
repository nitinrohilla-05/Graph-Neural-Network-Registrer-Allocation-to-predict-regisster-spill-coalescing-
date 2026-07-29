"""
End-to-End Integration Tests for Synthetic Generation, Dataset, Model Training, and Evaluation.
"""

import unittest
from dataset.generator import SyntheticIRGenerator
from dataset.dataset import InterferenceGraphDataset
from models.gnn_allocator import RelationalGNNRegisterAllocator
from models.trainer import GNNTrainer
from evaluation.evaluator import BenchmarkEvaluator


class TestEndToEndPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        # 1. Dataset generation
        generator = SyntheticIRGenerator(seed=123)
        raw_samples = generator.generate_dataset(num_samples=5, min_vars=5, max_vars=10, num_registers=4)
        self.assertEqual(len(raw_samples), 5)

        # 2. PyTorch Dataset
        dataset = InterferenceGraphDataset(raw_samples, num_registers=4)
        self.assertEqual(len(dataset), 5)

        # 3. Model & Trainer
        model = RelationalGNNRegisterAllocator(in_node_features=6, hidden_dim=16, num_registers=4)
        trainer = GNNTrainer(model, dataset, lr=1e-2)

        history = trainer.train(num_epochs=2, verbose=False)
        self.assertEqual(len(history["train_loss"]), 2)

        # 4. Benchmark evaluation
        evaluator = BenchmarkEvaluator(num_registers=4)
        results = evaluator.benchmark_batch(model, raw_samples)
        self.assertIn("GNN", results)
        self.assertIn("Chaitin-Briggs", results)


if __name__ == "__main__":
    unittest.main()
