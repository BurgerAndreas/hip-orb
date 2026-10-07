"""Shared message-edge ordering and independent, complete Hessian pair topology."""
import torch
from contextlib import contextmanager


@contextmanager
def deterministic_graph_audit():
    """Isolate topology checks from nondeterministic reductions and TF32 rounding."""
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def canonical_message_graph(graph):
    """Sort directed edges and every aligned edge tensor, without changing edges."""
    n = int(graph.batch_total_n_node)
    order = torch.argsort(graph.senders * n + graph.receivers, stable=True)
    graph.senders, graph.receivers = graph.senders[order], graph.receivers[order]
    for field in ("edge_features", "edge_targets"):
        values = getattr(graph, field)
        setattr(graph, field, {key: value[order] if value is not None else None
                               for key, value in values.items()})
    return graph


def refresh_message_vectors(graph):
    """Recompute molecular geometry on the inference device from final positions."""
    if bool(graph.system_features["pbc"].any()):
        raise NotImplementedError("CPU HIP graph preparation supports molecules only")
    if bool((graph.edge_features["unit_shifts"] != 0).any()):
        raise ValueError("Nonperiodic message edges have nonzero image shifts")
    graph.edge_features["vectors"] = (graph.node_features["positions"][graph.receivers]
                                      - graph.node_features["positions"][graph.senders])
    return graph


def hessian_pair_topology(senders, receivers, sizes):
    """CPU indices only. Latent features and pair geometry remain device computations."""
    if senders.device.type != "cpu" or receivers.device.type != "cpu":
        raise ValueError("Build CPU pair topology before transferring message edges")
    if senders.shape != receivers.shape or senders.ndim != 1:
        raise ValueError("Invalid directed message edge arrays")
    sizes = [int(n) for n in sizes]
    total = sum(sizes)
    if min(sizes, default=0) < 1 or bool(((senders < 0) | (senders >= total) |
                                       (receivers < 0) | (receivers >= total)).any()):
        raise ValueError("Invalid atom sizes or message endpoints")
    if len(torch.unique(senders * total + receivers)) != len(senders):
        raise ValueError("Duplicate directed message edges")
    results, offset = [], 0
    for n in sizes:
        mask = (senders >= offset) & (senders < offset + n)
        if bool(((receivers[mask] < offset) | (receivers[mask] >= offset + n)).any()):
            raise ValueError("Message edges cross molecules")
        mapping = torch.full((n, n), -1, dtype=torch.long)
        ids = torch.arange(len(senders))[mask]
        mapping[senders[mask] - offset, receivers[mask] - offset] = ids
        i, j = torch.triu_indices(n, n, offset=1)
        results.append({"i": i, "j": j, "forward": mapping[i, j], "reverse": mapping[j, i]})
        offset += n
    return results
