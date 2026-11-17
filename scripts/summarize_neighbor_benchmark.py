"""Summarize retained raw timing repeats without hiding failures or pair changes."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(directory):
    groups = defaultdict(list)
    errors = []
    for line in (directory / "per_structure.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["status"] != "ok":
            errors.append(row)
            continue
        groups[row["structure_id"], row["mode"]].append(row)
    results = []
    for (identity, mode), rows in groups.items():
        if len({row["pairs"] for row in rows}) != 1:
            raise ValueError(f"Backend pair counts differ for {identity}, {mode}")
        timings = {}
        for backend in sorted({row["backend"] for row in rows}):
            selected = [row for row in rows if row["backend"] == backend]
            raw = [row["seconds"] for row in selected]
            timings[backend] = {
                "median_seconds": statistics.median(raw),
                "min_seconds": min(raw),
                "max_seconds": max(raw),
                "repeats": len(raw),
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in selected),
            }
        results.append(
            {
                "structure_id": identity,
                "natoms": rows[0]["natoms"],
                "mode": mode,
                "pairs": rows[0]["pairs"],
                "timings": timings,
            }
        )
    results.sort(key=lambda x: (x["natoms"], x["mode"]))
    return {
        "definition": "median of synchronized wall seconds; all raw repeats remain in per_structure.jsonl",
        "results": results,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("atoms mode brute_s cpu_async_s gpu_s")
    for row in result["results"]:
        cells = [
            f"{row['timings'][name]['median_seconds']:.6f}" if name in row["timings"] else "missing"
            for name in ["brute", "cpu_async", "gpu"]
        ]
        print(row["natoms"], row["mode"], *cells)
    print(f"failed records: {len(result['errors'])}")


if __name__ == "__main__":
    main()
