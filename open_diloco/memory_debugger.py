from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Dict, Iterable, Iterator, List, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Tuple["TensorLike", ...], List["TensorLike"]]


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    if not isinstance(tensor, torch.Tensor):
        return 0
    return tensor.element_size() * tensor.numel()


def _collect_nested_bytes(obj: TensorLike) -> int:
    if isinstance(obj, torch.Tensor):
        return _tensor_nbytes(obj)
    if isinstance(obj, (tuple, list)):
        return sum(_collect_nested_bytes(item) for item in obj)
    return 0


def bytes_to_gb(value: int) -> float:
    return float(value) / (1024.0 ** 3)


class MemoryUsageTracker:
    """
    Runtime memory instrumentation helper.

    Tracks parameter, gradient, optimizer state and activation footprint for a PyTorch model.
    Activation footprint is measured by registering lightweight forward hooks on leaf modules.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | object | None = None):
        self.model = model
        self.optimizer = optimizer
        self.activation_bytes: int = 0
        self._activation_by_module: Dict[str, int] = defaultdict(int)
        self._leaf_modules: List[Tuple[str, torch.nn.Module]] = self._compute_leaf_modules()
        self._device = self._infer_device()

    def _infer_device(self) -> torch.device | None:
        for param in self.model.parameters():
            if param.device is not None:
                return param.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_leaf_modules(self) -> List[Tuple[str, torch.nn.Module]]:
        leaves: List[Tuple[str, torch.nn.Module]] = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                leaves.append((name, module))
        return leaves

    def parameter_bytes(self) -> int:
        total = 0
        for param in self.model.parameters():
            total += _tensor_nbytes(param.data)
        return total

    def gradient_bytes(self) -> int:
        total = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total += _tensor_nbytes(param.grad)
        return total

    def _iter_optimizer_tensors(self, optimizer: object) -> Iterator[torch.Tensor]:
        stack: List[object] = [optimizer]
        visited: set[int] = set()
        while stack:
            opt = stack.pop()
            if opt is None:
                continue
            opt_id = id(opt)
            if opt_id in visited:
                continue
            visited.add(opt_id)

            if isinstance(opt, torch.optim.Optimizer):
                for state in opt.state.values():
                    if isinstance(state, dict):
                        for value in state.values():
                            if isinstance(value, torch.Tensor):
                                yield value
            # Common wrappers
            inner = getattr(opt, "inner_optimizer", None)
            if inner is not None and id(inner) not in visited:
                stack.append(inner)
            state_averager = getattr(opt, "state_averager", None)
            if state_averager is not None:
                offload_opt = getattr(state_averager, "optimizer", None)
                if offload_opt is not None and id(offload_opt) not in visited:
                    stack.append(offload_opt)
            optimizers = getattr(opt, "optimizers", None)
            if isinstance(optimizers, Iterable):
                for nested in optimizers:
                    if nested is not None and id(nested) not in visited:
                        stack.append(nested)

    def optimizer_state_bytes(self) -> int:
        if self.optimizer is None:
            return 0
        total = 0
        for tensor in self._iter_optimizer_tensors(self.optimizer):
            total += _tensor_nbytes(tensor)
        return total

    def reset_activation_stats(self) -> None:
        self.activation_bytes = 0
        self._activation_by_module.clear()

    def _activation_hook(self, module_name: str, _mod, _inp, outp) -> None:
        bytes_used = _collect_nested_bytes(outp)
        if bytes_used <= 0:
            return
        self.activation_bytes += bytes_used
        self._activation_by_module[module_name] += bytes_used

    @contextmanager
    def capture_activations(self):
        """
        Context manager to instrument activations during a single forward pass.
        """
        self.reset_activation_stats()
        handles = []
        for name, module in self._leaf_modules:
            handle = module.register_forward_hook(partial(self._activation_hook, name))
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def activation_topk(self, k: int = 5) -> List[Tuple[str, int]]:
        if not self._activation_by_module:
            return []
        return sorted(self._activation_by_module.items(), key=lambda item: item[1], reverse=True)[:k]

    def device_allocated_bytes(self) -> int:
        if self._device is None or not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated(device=self._device)

    def device_reserved_bytes(self) -> int:
        if self._device is None or not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_reserved(device=self._device)

    def max_device_allocated_bytes(self) -> int:
        if self._device is None or not torch.cuda.is_available():
            return 0
        return torch.cuda.max_memory_allocated(device=self._device)

    def snapshot(self) -> Dict[str, int]:
        """
        Return the latest per-component memory usage in bytes.
        """
        return {
            "parameters_bytes": self.parameter_bytes(),
            "gradients_bytes": self.gradient_bytes(),
            "optimizer_bytes": self.optimizer_state_bytes(),
            "activations_bytes": self.activation_bytes,
            "cuda_allocated_bytes": self.device_allocated_bytes(),
            "cuda_reserved_bytes": self.device_reserved_bytes(),
            "cuda_max_allocated_bytes": self.max_device_allocated_bytes(),
        }


