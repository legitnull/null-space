"""
Example usage of debug hooks for model porting.

Run with: python -m null_space.hamster.demo.debug_hook_demo
"""

import torch
import torch.nn as nn

from null_space.hamster.debug import (
    DebugHooks,
    register_debug_hooks,
    remove_debug_hooks_force,
    diagnose_model,
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


def example_context_manager():
    """Example using DebugHooks as a context manager."""
    print("\n" + "=" * 60)
    print("Example 1: Context Manager Usage")
    print("=" * 60 + "\n")

    model = SimpleModel()
    x = torch.randn(2, 32, requires_grad=True)

    with DebugHooks(model):
        output = model(x)
        loss = output.sum()
        loss.backward()

    print("\nHooks automatically removed after context exits")


def example_functional_api():
    """Example using the functional API."""
    print("\n" + "=" * 60)
    print("Example 2: Functional API Usage")
    print("=" * 60 + "\n")

    model = SimpleModel()
    x = torch.randn(2, 32, requires_grad=True)

    # Register hooks
    hooks = register_debug_hooks(model)

    # Forward pass
    print("\n--- Forward Pass ---")
    output = model(x)

    # Backward pass
    print("\n--- Backward Pass ---")
    loss = output.sum()
    loss.backward()

    # Clean up
    hooks.remove()


def example_force_remove():
    """Example of force removing hooks without handles."""
    print("\n" + "=" * 60)
    print("Example 3: Force Remove Hooks")
    print("=" * 60 + "\n")

    model = SimpleModel()

    # Simulate losing hook handles (e.g., from another library)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module.register_forward_hook(lambda m, i, o: None)

    print(f"Hooks registered on model (no handles saved)")

    # Force remove all hooks
    remove_debug_hooks_force(model)


def example_comparing_frameworks():
    """
    Example showing how to use hooks for comparing two implementations.

    This is the primary use case: verifying numerical equivalence when
    porting a model from one framework to another.
    """
    print("\n" + "=" * 60)
    print("Example 4: Comparing Two Implementations")
    print("=" * 60 + "\n")

    # Simulating two different implementations
    torch.manual_seed(42)
    model_v1 = SimpleModel(hidden_size=64)

    torch.manual_seed(42)
    model_v2 = SimpleModel(hidden_size=64)

    # Same input
    torch.manual_seed(0)
    x = torch.randn(2, 32)

    # Collect outputs
    outputs = {}

    def make_collector(name, storage):
        def hook(module, input, output):
            key = f"{name}.output"
            storage[key] = output.detach().clone()
        return hook

    # Hook model_v1
    for name, module in model_v1.named_modules():
        if len(list(module.children())) == 0 and not isinstance(module, nn.Dropout):
            module.register_forward_hook(make_collector(f"v1.{name}", outputs))

    # Hook model_v2
    for name, module in model_v2.named_modules():
        if len(list(module.children())) == 0 and not isinstance(module, nn.Dropout):
            module.register_forward_hook(make_collector(f"v2.{name}", outputs))

    # Run both models
    out_v1 = model_v1(x)
    out_v2 = model_v2(x)

    # Compare outputs
    print("Comparing layer outputs:")
    for name in ["linear1", "activation", "linear2"]:
        v1_key = f"v1.{name}.output"
        v2_key = f"v2.{name}.output"
        diff = (outputs[v1_key] - outputs[v2_key]).abs().max().item()
        status = "MATCH" if diff < 1e-6 else "DIFF"
        print(f"  {name}: max_diff={diff:.2e} {status}")

    # Clean up
    remove_debug_hooks_force(model_v1)
    remove_debug_hooks_force(model_v2)


if __name__ == "__main__":
    example_context_manager()
    example_functional_api()
    example_force_remove()
    example_comparing_frameworks()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
