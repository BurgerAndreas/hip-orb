# Archived speed experiments

These experiments do not form part of the training package.

- `bounded_graphs.py` captures repeated frozen-backbone shapes. It produces no reuse in the tested ordered batches.
- `linear_backward.py` tests FP16 backward products for individual linear layers. Conversion overhead removes the speed gain.
- `check_bounded_graphs.py` retains the graph correctness check. Run it explicitly when you revisit this experiment.

The selected MLP backward implementation stays in `hip_orb/backward_precision.py`.
Original benchmark scripts, full test matrices, and results stay in [the round-three evidence](../../../cluster_runs/hip_training_speed_round3_20261021).
Immutable cluster runtimes and checkpoint continuations retain their verified code.
