# Checkpoint and stop on request

Start the original runner with `--stop-request-file /shared/run/STOP`. To pause,
create that file (`touch /shared/run/STOP`). The runner finishes any current
optimizer update, then uses its existing atomic `head.pt` save, records an atomic
`report.json` with `stop_reason: requested` and `auto_continue: false`, skips final
validation and closes loader workers. The file persists so requests cannot be lost.
Remove it before explicitly resuming with the same arguments and `--checkpoint`.
A request cannot interrupt a blocked loader, a GPU call or validation in progress.
No signal handler, background thread or checkpoint writer is added.

Adam, global step, epoch/batch cursor, batch-plan metadata and LR configuration
use the original save/resume path. New checkpoints also store the training seed
and CPU/current-device CUDA torch RNG state; changing the seed on resume is rejected. Old
checkpoints remain readable (they lack a stored RNG state). Loader worker seed
draws use a dedicated seeded generator so iterator recreation does not advance the
restored trainer RNG. Workers only read immutable payloads. Consistency seeds
continue to derive from the unchanged global step. Use the same data and config.

When a caller has initialized torch.distributed, matching update/loop boundaries reduce stop, session and total-limit flags
across ranks; intermediate bookkeeping queries reuse the agreed decision. A request visible only to one rank stops all ranks.
The caller must already provide synchronized optimizer updates, matching runner
control flow and separate output directories for each rank. This does not add a
distributed training implementation to the base runner. In particular, do not run
multiple independent base runners against one output directory. Existing timed
stops also use the coordinated decision when a process group is initialized.

`hip_train_steps.sbatch` honors requested stops and submits no automatic successor.
Other launchers must check `stop_reason` or `auto_continue`, including any custom
launcher that previously continued whenever `training_complete` was false.
The feature is new; historical MI100 deployment only supported timed sessions.
Production artifacts and completed training remain untouched.
