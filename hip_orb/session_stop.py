"""Stop decisions at runner update boundaries, optionally across initialized ranks."""
from pathlib import Path

import torch
import torch.distributed as dist


class UpdateBoundary:
    def __init__(self, request_file=None):
        self.request_file = None if request_file is None else Path(request_file)
        self.requested = False
        self.stopped = False
        self.total_limited = False

    def reached(self, local_limit, total_limit=False):
        self.requested |= self.request_file is not None and self.request_file.is_file()
        flags = [int(self.requested), int(local_limit), int(total_limit)]
        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
            value = torch.tensor(flags, device=device, dtype=torch.int32)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            flags = value.tolist()
        self.requested = bool(flags[0])
        self.total_limited |= bool(flags[2])
        self.stopped |= any(flags)
        return self.stopped
