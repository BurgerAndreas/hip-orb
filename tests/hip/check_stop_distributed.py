"""torchrun --nproc_per_node=2 (optional HIP_TEST_BACKEND=nccl), or --local-test."""
from datetime import timedelta
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hip_orb.session_stop import UpdateBoundary


def check(rank, world_size, init_method, backend):
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend, init_method=init_method, rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=60))
    try:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "STOP"
            boundary = UpdateBoundary(path)
            assert not boundary.reached(False)
            if rank == 1:
                path.touch()
            assert boundary.reached(False) and boundary.requested
            path.unlink(missing_ok=True)
            assert boundary.reached(False)
            limit = UpdateBoundary()
            assert limit.reached(rank == world_size - 1) and not limit.requested
            assert not UpdateBoundary().reached(False)
            total = UpdateBoundary()
            assert total.reached(False, total_limit=rank == world_size - 1)
            assert total.total_limited and not total.requested
        if rank == 0:
            print(f"STOP_CONSENSUS_PASS backend={backend} ranks={world_size}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--local-test" in sys.argv:
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(check, args=(3, f"file://{root}/rendezvous", "gloo"), nprocs=3)
    else:
        check(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), "env://",
              os.environ.get("HIP_TEST_BACKEND", "gloo"))
