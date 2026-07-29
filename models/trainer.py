"""
Training Pipeline Module for GNN Register Allocator.
Manages epoch optimization, metric computation, class imbalance weighting, and model checkpoint saving/loading.
"""

from typing import Dict, Any, List, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import InterferenceGraphDataset, GraphDataSample
from models.gnn_allocator import RelationalGNNRegisterAllocator, GraphColoringLoss


class GNNTrainer:
    """Trainer class for training and validating GNN Register Allocator models."""
    def __init__(
        self,
        model: nn.Module,
        dataset: Any,
        val_dataset: Optional[Any] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        use_pos_weight: bool = True
    ):
        self.model: nn.Module = model.to(device)
        self.dataset: Any = dataset
        self.val_dataset: Optional[Any] = val_dataset
        self.device: torch.device = torch.device(device)

        pos_weight = None
        if use_pos_weight and hasattr(dataset, "get_pos_weight"):
            pos_weight = dataset.get_pos_weight().to(self.device)

        self.criterion = GraphColoringLoss(conflict_weight=0.5, coalesce_weight=0.3, pos_weight=pos_weight)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "ce_loss": [],
            "conflict_loss": [],
            "val_accuracy": [],
            "val_color_violations": []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Trains for one epoch across all dataset samples."""
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_conflict = 0.0

        for sample in self.dataset:
            x = sample.node_features.to(self.device)
            interf_adj = sample.interf_adj.to(self.device)
            coal_adj = sample.coal_adj.to(self.device)
            target_colors = sample.target_colors.to(self.device)
            coalesce_labels = sample.coalesce_labels.to(self.device)

            self.optimizer.zero_grad()
            color_logits, coalesce_scores = self.model(x, interf_adj, coal_adj)

            loss, loss_dict = self.criterion(
                color_logits, coalesce_scores, target_colors, interf_adj, coal_adj, coalesce_labels
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_dict["total_loss"]
            total_ce += loss_dict["ce_loss"]
            total_conflict += loss_dict["conflict_loss"]

        self.scheduler.step()
        n_samples = max(len(self.dataset), 1)

        metrics = {
            "train_loss": total_loss / n_samples,
            "ce_loss": total_ce / n_samples,
            "conflict_loss": total_conflict / n_samples
        }

        self.history["train_loss"].append(metrics["train_loss"])
        self.history["ce_loss"].append(metrics["ce_loss"])
        self.history["conflict_loss"].append(metrics["conflict_loss"])

        return metrics

    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """Evaluates model performance: accuracy and color conflict rate."""
        self.model.eval()
        ds = eval_dataset or self.val_dataset or self.dataset

        correct_preds = 0
        total_preds = 0
        total_interf_edges = 0
        color_violations = 0

        num_registers = getattr(self.model, "num_registers", 4)

        with torch.no_grad():
            for sample in ds:
                x = sample.node_features.to(self.device)
                interf_adj = sample.interf_adj.to(self.device)
                coal_adj = sample.coal_adj.to(self.device)
                target_colors = sample.target_colors.to(self.device)

                color_logits, _ = self.model(x, interf_adj, coal_adj)
                pred_classes = torch.argmax(color_logits, dim=-1)  # [N]

                correct_preds += torch.sum(pred_classes == target_colors).item()
                total_preds += len(target_colors)

                N = len(pred_classes)
                for i in range(N):
                    for j in range(i + 1, N):
                        if interf_adj[i, j] > 0:
                            c_i, c_j = pred_classes[i].item(), pred_classes[j].item()
                            if c_i < num_registers and c_j < num_registers:
                                total_interf_edges += 1
                                if c_i == c_j:
                                    color_violations += 1

        accuracy = correct_preds / max(total_preds, 1)
        violation_rate = (color_violations / max(total_interf_edges, 1)) * 100.0

        return {
            "accuracy": accuracy,
            "color_violations": color_violations,
            "violation_rate_pct": violation_rate
        }

    def train(self, num_epochs: int = 15, verbose: bool = True) -> Dict[str, List[float]]:
        """Runs multi-epoch training loop."""
        if verbose:
            print(f"Starting GNN Training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()

            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_color_violations"].append(val_metrics["violation_rate_pct"])

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
                print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - "
                      f"Loss: {train_metrics['train_loss']:.4f} (CE: {train_metrics['ce_loss']:.4f}, Conflict: {train_metrics['conflict_loss']:.4f}) | "
                      f"Val Acc: {val_metrics['accuracy']*100:.1f}% | Color Violations: {val_metrics['violation_rate_pct']:.2f}%")

        return self.history

    def save_checkpoint(self, filepath: str = "gnn_allocator.pt"):
        """Saves model state dict."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_node_features": getattr(self.model, "in_channels", getattr(self.model, "in_node_features", 6)),
            "hidden_dim": getattr(self.model, "hidden_channels", getattr(self.model, "hidden_dim", 64)),
            "num_registers": getattr(self.model, "num_registers", 4)
        }, filepath)
        print(f"Saved GNN Model Checkpoint to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str = "gnn_allocator.pt", device: str = "cpu") -> RelationalGNNRegisterAllocator:
        """Loads GNN model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        model = RelationalGNNRegisterAllocator(
            in_node_features=checkpoint.get("in_node_features", 6),
            hidden_dim=checkpoint.get("hidden_dim", 64),
            num_registers=checkpoint.get("num_registers", 4)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
