import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule(optimizer: Optimizer, total_steps: int, warmup_steps: int = 0) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, multiplier)
