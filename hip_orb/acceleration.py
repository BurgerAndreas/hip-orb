"""Isolate verified execution changes for matched benchmarks and audits."""

from contextlib import ExitStack, contextmanager
from types import MethodType

from hip_orb.execution import configure_execution, edge_values, known_charge_forward


@contextmanager
def speed_candidate(model, name):
    """Restore the previous execution settings after a bounded comparison."""
    if "+" in name:
        with ExitStack() as stack:
            for part in name.split("+"):
                stack.enter_context(speed_candidate(model, part))
            yield
        return
    if name in ("eager", "production"):
        previous = getattr(model, "execution_config", {}).get("mode", "eager")
        try:
            configure_execution(model, "optimized" if name == "production" else "eager")
            yield
        finally:
            configure_execution(model, previous)
        return
    if name not in ("tensor", "masked_edges", "tensor_graph", "async_metadata"):
        raise ValueError(f"Unknown speed candidate: {name}")
    saved, methods = [], []

    def replace(obj, attribute, value):
        saved.append((obj, attribute, hasattr(obj, attribute), getattr(obj, attribute, None)))
        setattr(obj, attribute, value)

    try:
        if name == "tensor":
            replace(model.head, "_tensor_training", True)
        elif name == "masked_edges":
            replace(model.head, "_edge_gather", edge_values)
        elif name == "tensor_graph":
            replace(model, "_tensor_graph", True)
        else:
            from hip_orb.compilation import shape_known_conditioning
            from orb_models.forcefield.models.forcefield_heads import LatentChargeHead
            replace(model, "_async_metadata", True)
            conditioner = getattr(model.base.model, "conditioner", None)
            if conditioner is not None:
                methods.append((conditioner, conditioner.forward))
                conditioner.forward = MethodType(shape_known_conditioning, conditioner)
            charges = model.base.heads["latent_charges"]
            if isinstance(charges, LatentChargeHead):
                methods.append((charges, charges.forward))
                charges.forward = MethodType(known_charge_forward, charges)
        yield
    finally:
        for obj, attribute, existed, original in reversed(saved):
            if existed:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)
        for module, forward in reversed(methods):
            module.forward = forward
