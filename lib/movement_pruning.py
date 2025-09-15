# SPDX-License-Identifier: MIT

"""
Movement Pruning implementation based on the paper:
"Movement Pruning: Adaptive Sparsity by Fine-Tuning"
Victor Sanh, Thomas Wolf, Alexander M. Rush (2020)
https://arxiv.org/abs/2005.07683
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MovementPruning:
    """
    Implements Movement Pruning for adaptive sparsity during fine-tuning.

    Movement pruning learns the mask during training by tracking weight movements
    and pruning weights that move towards zero while keeping those that move away.
    """

    def __init__(
        self,
        model,
        initial_sparsity=0.0,
        target_sparsity=0.9,
        warmup_steps=0,
        pruning_frequency=100,
        ramp_end_step=3000,
    ):
        """
        Initialize Movement Pruning.

        Args:
            model: PyTorch model to prune
            initial_sparsity: Starting sparsity level (0 to 1)
            target_sparsity: Final target sparsity level (0 to 1)
            warmup_steps: Number of steps before pruning starts
            pruning_frequency: Update masks every N steps
            ramp_end_step: Step at which target sparsity is reached
        """
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.warmup_steps = warmup_steps
        self.pruning_frequency = pruning_frequency
        self.ramp_end_step = ramp_end_step
        self.step = 0

        # Initialize masks and scores for prunable layers
        self.masks = {}
        self.scores = {}
        self.initial_weights = {}

        # Get prunable layers (Conv2d and Linear layers)
        self.prunable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable_layers.append((name, module))

                # Initialize binary masks (1 = keep, 0 = prune)
                weight_shape = module.weight.shape
                self.masks[name] = torch.ones(
                    weight_shape, dtype=torch.float32, device=module.weight.device
                )

                # Initialize movement scores
                self.scores[name] = torch.zeros(
                    weight_shape, dtype=torch.float32, device=module.weight.device
                )

                # Store initial weights for movement calculation
                self.initial_weights[name] = module.weight.data.clone()

                # Register mask as buffer in the module
                module.register_buffer(f"pruning_mask", self.masks[name])

    def compute_movement_scores(self):
        """
        Compute movement scores S_i = W_i * (W_i - W_i^0)
        where W_i is current weight and W_i^0 is initial weight.

        Positive score: weight moving away from zero (keep)
        Negative score: weight moving towards zero (prune)
        """
        for name, module in self.prunable_layers:
            current_weight = module.weight.data
            initial_weight = self.initial_weights[name]

            # Movement score: weight * (weight - initial_weight)
            # This captures both magnitude and direction of movement
            movement = current_weight - initial_weight
            self.scores[name] = current_weight * movement

    def get_current_sparsity(self):
        """Calculate current sparsity level based on schedule."""
        if self.step < self.warmup_steps:
            return 0.0

        if self.step >= self.ramp_end_step:
            return self.target_sparsity

        # Linear ramp from initial to target sparsity
        ramp_progress = (self.step - self.warmup_steps) / (
            self.ramp_end_step - self.warmup_steps
        )
        current_sparsity = (
            self.initial_sparsity
            + (self.target_sparsity - self.initial_sparsity) * ramp_progress
        )

        return current_sparsity

    def update_masks(self, iter_num=None):
        """Update binary masks based on movement scores and target sparsity.

        Args:
            iter_num: Current iteration number (for compatibility with other pruners)
        """
        current_sparsity = self.get_current_sparsity()

        if current_sparsity == 0.0:
            return

        # Compute movement scores
        self.compute_movement_scores()

        # Collect all scores to determine global threshold
        all_scores = []
        for name in self.scores:
            all_scores.append(self.scores[name].flatten())

        all_scores = torch.cat(all_scores)

        # Find threshold for target sparsity
        # Prune weights with lowest movement scores
        k = int(current_sparsity * all_scores.numel())
        if k > 0:
            threshold = torch.kthvalue(all_scores, k).values

            # Update masks
            for name, module in self.prunable_layers:
                # Weights with scores below threshold are pruned
                self.masks[name] = (self.scores[name] > threshold).float()

                # Update module's mask buffer
                module.pruning_mask.data = self.masks[name]

    def apply_masks(self):
        """Apply binary masks to weights."""
        for name, module in self.prunable_layers:
            module.weight.data *= self.masks[name]

    def step_pruning(self):
        """Called at each training step to update pruning."""
        self.step += 1

        # Update masks at specified frequency
        if self.step % self.pruning_frequency == 0:
            self.update_masks()

        # Always apply masks to ensure pruned weights stay zero
        self.apply_masks()

    def get_sparsity(self):
        """Get overall sparsity of the model."""
        total_params = 0
        total_pruned = 0

        for name in self.masks:
            mask = self.masks[name]
            total_params += mask.numel()
            total_pruned += (mask == 0).sum().item()

        return total_pruned / total_params if total_params > 0 else 0.0

    def get_sparsity_stats(self):
        """Get current sparsity statistics for monitoring."""
        stats = {}
        total_params = 0
        total_pruned = 0

        for name, module in self.prunable_layers:
            mask = self.masks[name]
            num_params = mask.numel()
            num_pruned = (mask == 0).sum().item()

            total_params += num_params
            total_pruned += num_pruned

            stats[name] = {
                "sparsity": num_pruned / num_params,
                "pruned": num_pruned,
                "total": num_params,
            }

        stats["global"] = {
            "sparsity": total_pruned / total_params if total_params > 0 else 0,
            "pruned": total_pruned,
            "total": total_params,
            "target_sparsity": self.get_current_sparsity(),
        }

        return stats

    def prune_model_final(self):
        """
        Final pruning to permanently remove pruned weights.
        This converts sparse weights to actual smaller matrices where possible.
        Note: This is mainly for structured pruning scenarios.
        """
        # Apply final masks
        self.apply_masks()

        # Return final sparsity
        stats = self.get_sparsity_stats()
        return stats["global"]["sparsity"]


class PrunedForward:
    """
    Context manager to ensure masked weights during forward pass.
    """

    def __init__(self, pruning_module):
        self.pruning = pruning_module

    def __enter__(self):
        self.pruning.apply_masks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
