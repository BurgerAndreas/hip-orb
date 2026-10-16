"""Compare held-out validation metrics before and after execution changes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from hip_orb.benchmark_speed import head_options
from hip_orb.benchmark_training import model_hash
from hip_orb.data import OrbMolDataset, make_loader
from hip_orb.execution import EXECUTION_MODES, configure_execution
from hip_orb.model import HIPOrb
from hip_orb.runner import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--modes", choices=EXECUTION_MODES, nargs="+", default=list(EXECUTION_MODES))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32-high", "float32-highest"), default="float32-high")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(args.deterministic)
    config = json.loads(args.config.read_text())
    dataset = OrbMolDataset(args.validation_index, training_only=True)
    loader = make_loader(dataset, workers=2, batch_size=config["validation_batch_size"],
                         max_hessian_elements=config["validation_max_hessian_elements"],
                         seed=7, max_samples=args.samples, pin_memory=True)
    digest = hashlib.sha256()
    for batch in loader.batch_sampler:
        digest.update(np.asarray(batch, dtype=np.int64).tobytes())
        digest.update(b"\n")
    model = HIPOrb.pretrained("cuda", weights_path=args.weights, precision=args.precision, **head_options(config))
    checkpoint = torch.load(args.head_checkpoint, map_location="cuda", weights_only=True)
    if checkpoint["head_config"] != model.head.config:
        raise ValueError("Validation checkpoint architecture differs")
    model.head.load_state_dict(checkpoint["head"])
    frozen_hash, head_hash = model_hash(model.base), model_hash(model.head)
    report = {"complete": False, "checkpoint_step": checkpoint["global_step"],
              "validation_rows_sha256": digest.hexdigest(), "samples_requested": args.samples,
              "modes": [], "weights_updated": False, "precision": args.precision,
              "deterministic": args.deterministic}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for mode in args.modes:
            execution = configure_execution(model, mode)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = validate(model, loader, "cuda", report_samples=0)
            torch.cuda.synchronize()
            row = {"execution": execution, "elapsed_s": time.perf_counter() - start,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), **result}
            report["modes"].append(row)
            save()
            print(json.dumps({key: row[key] for key in ("execution", "elapsed_s", "validation_samples", "validation")}),
                  flush=True)
        reference = report["modes"][0]["validation"]
        for row in report["modes"]:
            row["metric_change_from_first_mode"] = {key: value - reference[key]
                                                     for key, value in row["validation"].items()}
        report["frozen_backbone_unchanged"] = model_hash(model.base) == frozen_hash
        report["head_unchanged"] = model_hash(model.head) == head_hash
        report["complete"] = True
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
        dataset.close()
        save()


if __name__ == "__main__":
    main()
