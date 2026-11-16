"""Two-GPU correctness smoke test; launch with torchrun."""
import os
import runpy
from pathlib import Path

import torch


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size >= 2, "Use at least two GPU ranks"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_device("cuda")
    root = Path(__file__).resolve().parent
    checks = runpy.run_path(str(root / "test_single_owner_broadcast.py"))
    for sizes in [(1,), (5,), (2, 3)]:
        for checkpoint in [False, True]:
            checks["test_complete_head_updates"](sizes, checkpoint)
    checks["test_multiple_owners_keep_original_indexing_and_empty_rows"]()
    torch.cuda.synchronize()
    print(f"HEAD_GPU_PASS rank={rank} device={torch.cuda.get_device_name()}", flush=True)
    torch.set_default_device("cpu")
    consensus = runpy.run_path(str(root / "check_stop_distributed.py"))
    consensus["check"](rank, world_size, "env://", "nccl")


if __name__ == "__main__":
    main()
