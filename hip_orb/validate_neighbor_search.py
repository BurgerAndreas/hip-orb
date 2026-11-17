"""Save full pair-set comparisons and raw cutoff-boundary differences."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from hip_orb.neighbor_search import AsyncCPUPairs, neighbor_pairs


def sorted_keys(pairs, n):
    i, j = pairs
    return (i.long() * n + j).sort().values


def difference(a, b):
    slots = torch.searchsorted(b, a)
    if not len(b):
        return a
    return a[(slots == len(b)) | (b[slots.clamp_max(len(b) - 1)] != a)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[10000, 100000])
    args = parser.parse_args()
    torch.set_num_threads(4)
    provenance = json.loads(args.provenance.read_text())
    cutoff = provenance["arguments"]["cutoff"]
    sources = [
        Path(__file__),
        Path(__file__).with_name("neighbor_search.py"),
        Path(__file__).with_name("neighbor_kernels.py"),
    ]
    version = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    for record in provenance["input_structures"]:
        if record["natoms"] not in args.counts:
            continue
        with np.load(record["path"]) as data:
            cpu = torch.from_numpy(data["positions"].copy())
        p, n = cpu.cuda(), len(cpu)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with AsyncCPUPairs(cpu, [n], cutoff) as work:
            pairs = work.result("cuda")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        reference = sorted_keys(pairs, n)
        del pairs
        rows = [
            {
                "backend": "cpu_async",
                "seconds": seconds,
                "pairs": len(reference),
                "missing_pairs": 0,
                "extra_pairs": 0,
            }
        ]
        for backend in ["gpu", "brute"]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            pairs = neighbor_pairs(p, [n], cutoff, backend=backend)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            actual = sorted_keys(pairs, n)
            del pairs
            extra, missing = difference(actual, reference), difference(reference, actual)
            details = []
            for key in torch.cat((extra[:32], missing[:32])).cpu().tolist():
                i, j = divmod(key, n)
                d = p[i] - p[j]
                details.append(
                    {
                        "i": i,
                        "j": j,
                        "distance_float64_angstrom": float((p[i].double() - p[j].double()).norm()),
                        "distance_norm_float32_angstrom": float(d.norm()),
                        "distance_explicit_float32_angstrom": float(
                            ((d[0] ** 2 + d[1] ** 2) + d[2] ** 2).sqrt()
                        ),
                        "distance_cdist_float32_angstrom": float(
                            torch.cdist(
                                p[i : i + 1],
                                p[j : j + 1],
                                compute_mode="donot_use_mm_for_euclid_dist",
                            )[0, 0]
                        ),
                    }
                )
            rows.append(
                {
                    "backend": backend,
                    "seconds": seconds,
                    "pairs": len(actual),
                    "missing_pairs": len(missing),
                    "extra_pairs": len(extra),
                    "boundary_differences": details,
                }
            )
            del actual, extra, missing
        for row in rows:
            row.update(
                structure_id=record["structure_id"],
                natoms=n,
                geometry_sha256=record["geometry_sha256"],
                method_version=version,
                cutoff_angstrom=cutoff,
                mode="radius_search_and_full_pair_set_validation",
                units={"seconds": "s", "distance": "angstrom"},
            )
            with args.output.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "natoms",
                            "backend",
                            "seconds",
                            "pairs",
                            "missing_pairs",
                            "extra_pairs",
                        ]
                    }
                ),
                flush=True,
            )
        del reference, p
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
