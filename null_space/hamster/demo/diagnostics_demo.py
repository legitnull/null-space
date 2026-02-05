"""
Example usage of model diagnostics for debugging during model porting.

Run with: python -m null_space.hamster.demo.diagnostics_demo
"""

import torch
import torch.nn as nn

from null_space.hamster.debug import (
    diagnose_model,
    print_training_mode,
    print_frozen_params,
    print_device,
    print_dtype,
)


class SimpleModel(nn.Module):
    """A simple model for demonstration."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(32, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 16)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


def example_full_diagnostics():
    """Run full diagnostics on a model."""
    print("\n" + "=" * 60)
    print("Example 1: Full Model Diagnostics")
    print("=" * 60 + "\n")

    model = SimpleModel()
    diagnose_model(model)


def example_mixed_training_mode():
    """Detect modules accidentally left in eval mode."""
    print("\n" + "=" * 60)
    print("Example 2: Mixed Training/Eval Mode")
    print("=" * 60 + "\n")

    model = SimpleModel()
    model.train()

    # Accidentally leave one module in eval mode
    model.linear2.eval()

    print_training_mode(model)


def example_frozen_params():
    """Detect accidentally frozen parameters."""
    print("\n" + "=" * 60)
    print("Example 3: Frozen Parameters")
    print("=" * 60 + "\n")

    model = SimpleModel()

    # Accidentally freeze some parameters
    for param in model.linear1.parameters():
        param.requires_grad = False

    print_frozen_params(model)


def example_device_dtype():
    """Check device and dtype of parameters."""
    print("\n" + "=" * 60)
    print("Example 4: Device and Dtype")
    print("=" * 60 + "\n")

    model = SimpleModel()

    print_device(model)
    print()
    print_dtype(model)


if __name__ == "__main__":
    example_full_diagnostics()
    example_mixed_training_mode()
    example_frozen_params()
    example_device_dtype()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
