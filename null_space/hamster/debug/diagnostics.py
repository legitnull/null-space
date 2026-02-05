"""
Model diagnostics for debugging during model porting.

This module provides utilities to check model state and catch common
misconfigurations that can cause silent training failures.
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import defaultdict


def get_rank() -> int:
    """Get the current distributed rank, or 0 if not in distributed mode."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _format_size(num_params: int) -> str:
    """Format parameter count with human-readable suffix."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def print_training_mode(model: nn.Module) -> Dict[str, bool]:
    """
    Print training/eval mode of all modules (recursively).

    Args:
        model: PyTorch model

    Returns:
        Dict mapping module names to their training mode
    """
    rank = get_rank()

    by_mode = defaultdict(list)
    for name, module in model.named_modules():
        mode = "training" if module.training else "eval"
        by_mode[mode].append(name or "root")

    print(f"[Rank {rank}] Module training/eval mode:", flush=True)
    print(f"  training: {len(by_mode['training'])} modules", flush=True)
    print(f"  eval: {len(by_mode['eval'])} modules", flush=True)

    if by_mode["eval"] and by_mode["training"]:
        print(f"\n[Rank {rank}] WARNING: Mixed modes! Modules in eval:", flush=True)
        for name in by_mode["eval"]:
            print(f"    {name}", flush=True)

    return {name: module.training for name, module in model.named_modules()}


def print_frozen_params(model: nn.Module) -> Dict[str, bool]:
    """
    Print frozen parameters (requires_grad=False) recursively.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to requires_grad status
    """
    rank = get_rank()

    frozen = []
    trainable = []
    frozen_size = 0
    trainable_size = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
            trainable_size += param.numel()
        else:
            frozen.append(name)
            frozen_size += param.numel()

    total_size = frozen_size + trainable_size

    print(f"[Rank {rank}] Parameter requires_grad status:", flush=True)
    print(f"  trainable: {len(trainable)} params ({_format_size(trainable_size)})", flush=True)
    print(f"  frozen: {len(frozen)} params ({_format_size(frozen_size)})", flush=True)
    print(f"  total: {len(trainable) + len(frozen)} params ({_format_size(total_size)})", flush=True)

    if frozen:
        print(f"\n[Rank {rank}] Frozen parameters:", flush=True)
        for name in frozen:
            print(f"    {name}", flush=True)

    return {name: param.requires_grad for name, param in model.named_parameters()}


def print_device(model: nn.Module) -> Dict[str, torch.device]:
    """
    Print device of all parameters (recursively).

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to their device
    """
    rank = get_rank()

    by_device = defaultdict(list)
    devices = {}

    for name, param in model.named_parameters():
        devices[name] = param.device
        by_device[str(param.device)].append(name)

    print(f"[Rank {rank}] Parameter devices:", flush=True)
    for device, names in by_device.items():
        print(f"  {device}: {len(names)} params", flush=True)

    return devices


def print_dtype(model: nn.Module) -> Dict[str, torch.dtype]:
    """
    Print dtype of all parameters (recursively).

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to their dtype
    """
    rank = get_rank()

    by_dtype = defaultdict(list)
    dtypes = {}

    for name, param in model.named_parameters():
        dtypes[name] = param.dtype
        by_dtype[str(param.dtype)].append(name)

    print(f"[Rank {rank}] Parameter dtypes:", flush=True)
    for dtype, names in by_dtype.items():
        print(f"  {dtype}: {len(names)} params", flush=True)

    return dtypes


def diagnose_model(model: nn.Module) -> Dict:
    """
    Run all diagnostics on a model (recursively traverses all nested modules).

    Args:
        model: PyTorch model

    Returns:
        Dict with all diagnostic results
    """
    rank = get_rank()

    print(f"\n[Rank {rank}] " + "=" * 50, flush=True)
    print(f"[Rank {rank}] Model Diagnostics", flush=True)
    print(f"[Rank {rank}] " + "=" * 50 + "\n", flush=True)

    results = {
        "training_mode": print_training_mode(model),
        "requires_grad": print_frozen_params(model),
        "device": print_device(model),
        "dtype": print_dtype(model),
    }

    print(f"\n[Rank {rank}] " + "=" * 50 + "\n", flush=True)

    return results
