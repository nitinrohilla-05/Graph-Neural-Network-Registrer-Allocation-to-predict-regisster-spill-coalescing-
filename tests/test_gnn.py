"""
Unit Tests for Relational GNN Model Architecture and Loss Function.
"""

import unittest
import torch
from models.gnn_allocator import RelationalGNNRegisterAllocator, GraphColoringLoss


class TestGNNModel(unittest.TestCase):
    def test_forward_pass(self):
        N = 5
        K = 4
        in_dim = 6
        hidden_dim = 16

        model = RelationalGNNRegisterAllocator(in_node_features=in_dim, hidden_dim=hidden_dim, num_registers=K)

        x = torch.randn(N, in_dim)
        interf_adj = torch.eye(N)
        coal_adj = torch.zeros(N, N)

        color_logits, coalesce_scores = model(x, interf_adj, coal_adj)

        self.assertEqual(color_logits.shape, (N, K + 1))
        self.assertEqual(coalesce_scores.shape, (N, N))

    def test_loss_function(self):
        N = 4
        K = 4
        criterion = GraphColoringLoss()

        color_logits = torch.randn(N, K + 1)
        coalesce_scores = torch.sigmoid(torch.randn(N, N))
        target_colors = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        interf_adj = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float32)
        coal_adj = torch.zeros((N, N), dtype=torch.float32)
        coalesce_labels = torch.zeros((N, N), dtype=torch.float32)

        loss, loss_dict = criterion(color_logits, coalesce_scores, target_colors, interf_adj, coal_adj, coalesce_labels)

        self.assertTrue(loss.item() > 0)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("conflict_loss", loss_dict)


if __name__ == "__main__":
    unittest.main()
