"""Bounded CUDA graph capture for repeated frozen-backbone input shapes."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from types import MethodType

import torch


def _signature(value):
    if isinstance(value, torch.Tensor):
        metadata = (str(value.device), str(value.dtype), tuple(value.shape), tuple(value.stride()))
        return metadata if value.is_cuda else metadata + (tuple(value.flatten().tolist()),)
    if isinstance(value, dict):
        return tuple((key, _signature(v)) for key, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_signature(v) for v in value)
    return value


def _clone(value, targets, sources):
    if isinstance(value, torch.Tensor):
        out = value.clone()
        if value.is_cuda:
            targets.append(out)
            sources.append(value)
        return out
    if isinstance(value, dict):
        return {k: _clone(v, targets, sources) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_clone(v, targets, sources) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone(v, targets, sources) for v in value)
    return value


def _cuda_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value] if value.is_cuda else []
    if isinstance(value, dict):
        return [t for _, v in sorted(value.items()) for t in _cuda_tensors(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _cuda_tensors(v)]
    return []


@contextmanager
def frozen_backbone_graphs(
    model, *, max_graphs=4, max_nodes=1024, max_edges=32768, capture_after=3, max_attempts=8
):
    """Reuse frequent shapes without changing or padding any training batch."""
    original = model.base.model.forward
    cache, frequencies = OrderedDict(), OrderedDict()
    stats = {
        "captures": 0,
        "replays": 0,
        "eager": 0,
        "failures": [],
        "retained_input_bytes": 0,
        "attempts": 0,
    }
    excluded = hasattr(model.head, "backbone_mixture")
    try:

        def forward(self, batch):
            if (
                excluded
                or torch.is_grad_enabled()
                or self.training
                or len(batch.node_features["positions"]) > max_nodes
                or len(getattr(batch, "senders", ())) > max_edges
            ):
                stats["eager"] += 1
                return original(batch)
            fields = vars(batch)
            key = _signature(fields)
            if key in cache:
                graph, targets, output = cache[key]
                torch._foreach_copy_(targets, _cuda_tensors(fields))
                graph.replay()
                stats["replays"] += 1
                return output
            if len(cache) >= max_graphs or stats["attempts"] >= max_attempts:
                stats["eager"] += 1
                return original(batch)
            frequencies[key] = frequencies.pop(key, 0) + 1
            if len(frequencies) > 64:
                frequencies.popitem(last=False)
            if frequencies[key] != capture_after:
                stats["eager"] += 1
                return original(batch)
            stats["attempts"] += 1
            static = copy(batch)
            targets, sources = [], []
            for name, value in sorted(fields.items()):
                setattr(static, name, _clone(value, targets, sources))
            current = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    original(static)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    output = original(static)
            except RuntimeError as error:
                stats["failures"].append(str(error))
                stats["eager"] += 1
                return original(batch)
            cache[key] = graph, targets, output
            stats["captures"] += 1
            stats["retained_input_bytes"] += sum(t.numel() * t.element_size() for t in targets)
            # Capture records operations. Replay computes the first output too.
            graph.replay()
            return output

        model.base.model.forward = MethodType(forward, model.base.model)
        yield stats
    finally:
        model.base.model.forward = original
        cache.clear()
