# Single-owner affine conditioning

Enable with `--single-owner-broadcast` or `model.head.single_owner_broadcast = True`.
The default remains indexed conditioning. Multiple-molecule calls retain indexing;
one-molecule calls expand the existing scale/shift row. Parameters, state-dict names,
architecture config and precision are unchanged. Reduction order can change in FP32.

Provenance: CSLab MI100 study jobs 249443 and 249449 (15 November 2026).
Two repeats of 120 measured optimizer updates on three ranks measured 0.6403 to
0.8430 updates/s (+31.6%). All ranks passed losses, gradients, head weights and
Adam-state checks at atol=1e-6, rtol=1e-4, with unchanged frozen backbone hashes.
These compute timings exclude loader waits, runner bookkeeping and production
checkpoint I/O; they are not end-to-end production throughput. The original study
used immutable runtimes plus uncommitted changes, so base 83e5156 alone does not
reproduce it. This PR ports only the conditioning-row operation and validates it
against this base. No topology reuse, precision or compiler experiments are added.
