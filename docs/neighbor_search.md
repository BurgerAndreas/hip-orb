# Cutoff Hessian pair search

`hip_orb.neighbor_search.neighbor_pairs` returns int32 atom indices `(i, j)`
with `i < j`, in the same molecule, and distance strictly below the cutoff.
There is no neighbour cap. Pair order depends on the backend; align training
targets by atom indices, not by output position.

Backends are `brute` (tiled distances), `cpu` (SciPy KD-tree), and `gpu`
(sorted occupied cells and tiled Triton kernels). `auto` uses brute force for
at most 512 atoms, then GPU cells on CUDA or the CPU tree elsewhere. The GPU
index stores occupied cells only. CPU enumeration switches to bounded row
queries when the native pair array would exceed 4 GiB.

Coordinates and cutoff comparisons use float32. CPU and GPU use the same
three-coordinate reduction order as the existing CUDA distance calculation:
`sqrt((dx*dx + dz*dz) + dy*dy) < cutoff`. Conservative candidate searches and
a final predicate preserve strict membership at the boundary. No approximate
tree search is used. This order matters for distances within one float32 ULP
of the cutoff. Search coordinates are detached; the head recomputes geometry.

## Inference

```python
from hip_orb.cutoff_inference import cutoff_hessian

result = cutoff_hessian(model, graph, cutoff=30, pair_chunk=65536, search="gpu")
# result: i, j, off-diagonal 3x3 blocks, and N diagonal 3x3 blocks
```

For CPU overlap, submit the final CPU coordinates before building the GPU
graph. The graph must use the same atom order, coordinates, and molecule sizes.

```python
from hip_orb.neighbor_search import AsyncCPUPairs

with AsyncCPUPairs(cpu_positions, sizes, 30) as pending:
    graph = model.adapter.from_ase_atoms_list(molecules, device="cuda")
    result = cutoff_hessian(model, graph, 30, pair_chunk=65536, pair_future=pending)
```

`cutoff_forward` supports `block_sink(i, j, blocks)` under `torch.no_grad()`.
With a sink and no CPU future, it reuses a spatial plan and generates bounded
pair chunks twice: once for context pooling and once for readout. It returns
`diagonal` and `pairs`; the sink owns block storage. This bounds search and
head output memory without a dense matrix or a complete retained pair list.
Full output still requires 44 bytes per undirected pair for two int32 indices
and nine float32 values, before temporary buffers.

## Training and backbone memory

`cutoff_forward` is differentiable. `checkpoint_chunks=True` recomputes pair
MLPs in backward to reduce saved activations. The topology is discrete and is
not differentiated. The existing dense training runner keeps its current
objective. Sparse training callers must supply targets for the retained pairs
and diagonals; the cutoff path also truncates analytical Coulomb blocks.

For large graphs, `bounded_backbone_mlps(model.base.model, rows=65536)` is an
optional context manager. It chunks only the frozen backbone's independent
encoder and interaction MLP rows. Attention and graph reductions use the full
graph. Parameters, state keys, and conditioning stay unchanged. The backbone
must be frozen and in eval mode. GEMM row shapes can change float rounding.

## Measurement

Run `python -m hip_orb.benchmark_neighbor_search --help`, or use
`scripts/hip_neighbor_benchmark.sbatch`. Each run saves raw per-structure and
per-repeat wall times, GPU memory, pair counts, output checksums, input arrays,
source hashes, checkpoint hash, units, and the timing boundary.

The boundary includes graph construction, transfer, the frozen backbone,
search, and the Hessian head. A training step also includes loss, backward,
and AdamW. Source reads, model loading, and warmup are excluded. Training uses
zero block targets to measure execution; it does not measure model accuracy or
the full production data loader and rotation-consistency objective.

`validate_neighbor_search` compares complete pair sets, including the large
structure. `validate_backbone_chunks` saves real-checkpoint feature and block
differences. The benchmark uses prefixes of PDB 1AON; sizes beyond its atom
count use translated copies and are marked as scaling proxies.
