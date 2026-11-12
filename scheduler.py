"""
Learning Rate Scheduler — Noam Schedule
========================================
Section 5.3 of "Attention Is All You Need".

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

This corresponds to linearly increasing the learning rate for the first
warmup_steps training steps, and decreasing it thereafter proportionally
to the inverse square root of the step number.
"""

import torch


class NoamScheduler:
    """
    Implements the Noam learning rate schedule from the paper.

    Args:
        optimizer: PyTorch optimizer.
        d_model  (int): Model dimension.
        warmup_steps (int): Number of warmup steps.
        factor (float): Scaling factor (default 1.0).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0.0

    def step(self):
        """Update the learning rate."""
        self._step += 1
        rate = self._get_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate

    def _get_rate(self) -> float:
        """
        Compute the learning rate at the current step.

        lrate = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        """
        step = self._step
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def get_last_lr(self) -> float:
        return self._rate
