from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Tuple["TensorLike", ...], List["TensorLike"], Dict[str, "TensorLike"]]


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
    if isinstance(obj, dict):
        return sum(_collect_nested_bytes(item) for item in obj.values())
    return 0


def _extract_first_tensor(obj: TensorLike) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(obj, dict):
        for item in obj.values():
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return None


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
        self._activation_metadata: Dict[str, Tuple[Optional[torch.dtype], Optional[Tuple[int, ...]]]] = {}
        self._leaf_modules: List[Tuple[str, torch.nn.Module]] = self._compute_leaf_modules()
        self._device = self._infer_device()
        self._last_optimizer_dtype_bytes: Dict[torch.dtype, int] = {}
        self._last_optimizer_dtype_device_bytes: Dict[Tuple[str, str], int] = {}

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

    def parameter_stats(self) -> Tuple[int, int, int]:
        total = 0
        tensor_count = 0
        element_count = 0
        for param in self.model.parameters():
            bytes_used = _tensor_nbytes(param.data)
            total += bytes_used
            tensor_count += 1
            element_count += param.data.numel()
        return total, tensor_count, element_count

    def parameter_bytes(self) -> int:
        total, _, _ = self.parameter_stats()
        return total

    def parameter_tensor_count(self) -> int:
        _, tensor_count, _ = self.parameter_stats()
        return tensor_count

    def parameter_element_count(self) -> int:
        _, _, element_count = self.parameter_stats()
        return element_count

    def gradient_stats(self) -> Tuple[int, int, int]:
        total = 0
        tensor_count = 0
        element_count = 0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            bytes_used = _tensor_nbytes(param.grad)
            total += bytes_used
            tensor_count += 1
            element_count += param.grad.numel()
        return total, tensor_count, element_count

    def gradient_bytes(self) -> int:
        total, _, _ = self.gradient_stats()
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
        total = 0
        for tensor in self._iter_optimizer_tensors(self.optimizer) if self.optimizer is not None else []:
            total += _tensor_nbytes(tensor)
        return total

    def optimizer_state_stats(self) -> tuple[int, Dict[torch.dtype, int]]:
        if self.optimizer is None:
            return 0, {}
        total = 0
        dtype_bytes: Dict[torch.dtype, int] = defaultdict(int)
        dtype_device_bytes: Dict[Tuple[str, str], int] = defaultdict(int)
        for tensor in self._iter_optimizer_tensors(self.optimizer):
            bytes_used = _tensor_nbytes(tensor)
            if bytes_used <= 0:
                continue
            total += bytes_used
            dtype_bytes[tensor.dtype] += bytes_used
            device_type = tensor.device.type if tensor.device is not None else "unknown"
            dtype_device_bytes[(str(tensor.dtype), device_type)] += bytes_used
        self._last_optimizer_dtype_device_bytes = dtype_device_bytes
        return total, dtype_bytes

    def reset_activation_stats(self) -> None:
        self.activation_bytes = 0
        self._activation_by_module.clear()
        self._activation_metadata.clear()

    def _activation_hook(self, module_name: str, _mod, _inp, outp) -> None:
        bytes_used = _collect_nested_bytes(outp)
        if bytes_used <= 0:
            return
        self.activation_bytes += bytes_used
        self._activation_by_module[module_name] += bytes_used
        tensor = _extract_first_tensor(outp)
        dtype: Optional[torch.dtype] = None
        shape: Optional[Tuple[int, ...]] = None
        if tensor is not None:
            dtype = tensor.dtype
            shape = tuple(int(dim) for dim in tensor.shape)
        self._activation_metadata[module_name] = (dtype, shape)

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

    def activation_topk(self, k: int = 5) -> List[Tuple[str, int, Optional[torch.dtype], Optional[Tuple[int, ...]]]]:
        if not self._activation_by_module:
            return []
        sorted_items = sorted(self._activation_by_module.items(), key=lambda item: item[1], reverse=True)[:k]
        detailed = []
        for name, bytes_used in sorted_items:
            dtype, shape = self._activation_metadata.get(name, (None, None))
            detailed.append((name, bytes_used, dtype, shape))
        return detailed

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
        parameters_bytes, parameters_tensor_count, parameters_element_count = self.parameter_stats()
        gradients_bytes, gradients_tensor_count, gradients_element_count = self.gradient_stats()
        optimizer_bytes, optimizer_dtype_bytes = self.optimizer_state_stats()
        self._last_optimizer_dtype_bytes = optimizer_dtype_bytes
        return {
            "parameters_bytes": parameters_bytes,
            "parameters_tensor_count": parameters_tensor_count,
            "parameters_element_count": parameters_element_count,
            "gradients_bytes": gradients_bytes,
            "gradients_tensor_count": gradients_tensor_count,
            "gradients_element_count": gradients_element_count,
            "optimizer_bytes": optimizer_bytes,
            "activations_bytes": self.activation_bytes,
            "cuda_allocated_bytes": self.device_allocated_bytes(),
            "cuda_reserved_bytes": self.device_reserved_bytes(),
            "cuda_max_allocated_bytes": self.max_device_allocated_bytes(),
        }

    def optimizer_dtype_breakdown(self) -> Dict[str, float]:
        return {str(dtype): bytes_to_gb(bytes_used) for dtype, bytes_used in self._last_optimizer_dtype_bytes.items()}

    def optimizer_dtype_device_breakdown(self) -> Dict[str, float]:
        return {
            f"{dtype}|{device}": bytes_to_gb(bytes_used)
            for (dtype, device), bytes_used in self._last_optimizer_dtype_device_bytes.items()
        }


