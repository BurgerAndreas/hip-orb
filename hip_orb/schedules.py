"""Learning rates indexed by the absolute optimizer update number."""
import math


def learning_rate(step, peak_lr, schedule="constant", total_steps=None, warmup_fraction=0.15):
    if schedule == "constant":
        return peak_lr
    if schedule != "cosine" or not total_steps or not 0 < warmup_fraction < 1:
        raise ValueError("Cosine schedule requires total steps and a warmup fraction between zero and one")
    warmup = max(1, round(total_steps * warmup_fraction))
    if not 1 <= step <= total_steps or warmup >= total_steps:
        raise ValueError("Invalid schedule step or warmup length")
    if step <= warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
