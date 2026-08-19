# Resume review — consolidated findings (`feat/checkpoint-resume`)

Three review rounds, merged. **Round 1** read `git diff main...HEAD` plus the working tree and
hunted for bugs directly (formerly `report.md`). **Round 2** put four reviewers on the branch in
parallel, each emulating a kind of user rather than reading the diff: an **mlr3 power user**
(resample/benchmark/tuning/parallel/marshal), a **cluster operator** (the kill-and-restart loop), a
**config-fiddler** (change one setting between the two runs), and a **first-time reader** (check
every documented claim by running it). **Round 3** put three more on the callbacks specifically —
combinations of them, each stateful builtin in depth, and user-written ones plus the surrounding
workflow — under the rule that every parameter except `epochs` is identical between the two runs,
so nothing it reports depends on breaking the documented contract. Everything below was verified by
running code; each reviewer was timeboxed, so the coverage gaps at the end are real.

Finding ids are stable: `R1-n` from round 1, `R2-n` from round 2, `R3-n` from round 3. Resolved ones keep their id and
live in *Already addressed* at the bottom. Round 1 ran while the learner parameter was renamed from
`path` to `resume`; its findings are independent of the rename and all repros below use `resume`.

## Open findings

| # | Severity | Finding | Area |
|---|---|---|---|
| R2-2 | medium | Two concurrent runs share a checkpoint folder undetected (partially addressed) | guards |
| R2-8 | medium | A *corrupted* newest checkpoint bricks the folder, with no fallback | operability |
| R2-9 | medium | `resume = A` while checkpointing into B works exactly once | operability |
| R2-11 | medium | The documented per-fit-folder idiom breaks under `multisession` | ergonomics |
| R3-2 | medium | A `torch` tensor in a callback state is destroyed by the checkpoint | callback state |
| R3-1 | low | With `eval_freq > 1` the checkpoint epoch is validated an extra time | correctness |
| R3-3 | low | Whether an `on_end` state change reaches the checkpoint depends on `freq` | callback state |
| R2-13 | low | An architecture change with matching shapes resumes silently | guards |
| R2-14 | low | The checkpoint folder's writability is never checked | operability |
| R3-4 | low | A resumed `history` loses its `epoch` key | callback state |
| R3-5 | low | `ctx$step` is not restored, unlike the other counters | correctness |
| R3-6 | low | A folder holding another callback's file gives an opaque checkmate error | operability |
| R3-7 | low | An error inside a user `load_state_dict()` says nothing about the callback | operability |
| R3-8 | low | `on_begin`/`on_end` run once per run, so state accumulated there grows | docs |

---

## R2-2. MEDIUM — two concurrent runs share a checkpoint folder undetected (partially addressed)

The folder is read exactly twice, both times before any training: `can_checkpoint_into()` in
`initialize()` and the clash check in `on_begin()`. Two processes that start into the same empty
folder both see it empty, both pass, then interleave their writes. Last writer wins per file, so
`network<n>.pt`, `optimizer<n>.pt` and `state<n>.rds` for the same `n` can come from different
runs — and a later resume then loads a network from one run with an optimizer and callback states
from another. Both processes exit 0.

```r
w = function(shared, sd) { pkgload::load_all(".", quiet = TRUE)
  l = lrn("classif.mlp", epochs = 8, batch_size = 10, neurons = 50, seed = sd,
    callbacks = t_clbk("checkpoint", path = shared, freq = 1)); l$train(tsk("iris")) }
shared = tempfile()
p1 = callr::r_bg(w, list(shared, 1L)); p2 = callr::r_bg(w, list(shared, 2L)); p1$wait(); p2$wait()
c(p1$get_exit_status(), p2$get_exit_status())   # 0 0
```

Sequentially the guard *does* fire, which is what makes this dangerous — the protection looks
complete.

**Partially fixed.** `.save()` now checks all three files immediately before writing any of them and
errors if one is already there; the only exception is a checkpoint that was already half-written
when this run started, which `on_begin()` records as the one thing this run may complete. A second
run writing into the folder therefore errors at its next checkpoint instead of interleaving, and
the folder is left as it was rather than half rewritten. Covered by
*"a file that appears while a run is training is not written over"* and *"two fits of a `resample()`
cannot share a checkpoint folder"* in `test_CallbackSetCheckpoint.R`.

**What remains.** The check narrows the window rather than closing it: two processes that pass it
within the same instant still both write, `torch_save()` is not atomic so a reader can still see a
torn file, and `run.rds` has the same check-then-write race one level down. Closing it properly
needs atomic ownership of the folder (`dir.create()` returning `FALSE` is a reliable "someone else
has it", or an owner id in `run.rds` re-checked before each write) plus writing to a temp name and
`file.rename()`ing into place. Note that a hard lock has to be breakable: a run killed by SIGKILL
leaves one behind, and restarting after such a kill is the use case this feature exists for.

## R2-8. MEDIUM — a corrupted newest checkpoint bricks the folder, with no fallback

`checkpoint_files()` decides completeness by `file.exists()` alone. A file that exists but is
unreadable — the normal outcome of a power loss or a full disk — kills every later run inside
`torch_load`/`readRDS` with a message naming neither the file, nor the checkpoint, nor a remedy, and
there is no fallback to the older complete checkpoints sitting next to it. Eleven good checkpoints
are dead because of the twelfth.

```r
writeBin(raw(10), file.path(p, "network3.pt"))     # or state3.rds
lrn("classif.mlp", epochs = 6, batch_size = 50, neurons = 5, resume = p)$train(tsk("iris"))
# ERROR: parse error: premature EOF     (state3.rds -> "unknown input format")
```

**Fix:** have `latest_checkpoint()` fall back to the next-older complete checkpoint on a read
failure and warn; failing that, wrap the three loads so the error names the checkpoint and says it
can be deleted.

## R2-9. MEDIUM — `resume = A` while checkpointing into B works exactly once

Keeping the original folder read-only and checkpointing the continuation elsewhere is a natural
operator move. On the second run `.start_epoch` comes from A, so every epoch this run plans
collides with what it wrote into B, and `on_begin()` refuses — at any `epochs`. The message
("written by another run … continue that run instead of starting it over") is wrong here: the
clashing checkpoints were written by this same script, and the fix is to point `resume` at B.

**Fix:** when the clashing epochs in the write folder continue the same lineage, say so —
"'<B>' already continues this run up to epoch 6; set `resume` to '<B>'".

## R2-11. MEDIUM — the documented per-fit-folder idiom breaks under `multisession`

A `path` function written the obvious way keeps `globalenv()` as its enclosure, which is not
serialised to workers: `Error in path() : object 'i' not found`. Wrapping it in `local()` fixes
serialisation but gives each worker its own counter, so workers collide on folder names —
sequentially that errors, concurrently it is R2-2.

**Fix:** document that the function must be self-contained and must not depend on process-local
state; the safe recipes are `function() tempfile()` or a folder keyed on something globally unique.

Related observation from round 3: because the function takes no arguments and is called from
`initialize()`, it cannot key the folder on the fit it belongs to, so the safe recipes are also the
ones that make a `resample()`'s folders unresumable — `function() tempfile()` gives a fresh folder
on the next run, and `resume = TRUE` then silently starts from scratch. The documented promise
(unique folders, readable back off the model) does hold; resuming a *specific fit* does not follow
from it.

## R3-1. LOW — with `eval_freq > 1` the checkpoint epoch is validated an extra time

Lowered from medium and left as it is: judged not worth the wording it would take. Documentation
for it was written and removed again, so this section is the only record.

`eval_valid_in_epoch()` forces an evaluation in the last epoch of a run (`ctx$epoch ==
ctx$total_epochs`). For the run that writes the checkpoint that last epoch is the checkpoint epoch,
which the uninterrupted run does not treat specially — so when the checkpoint epoch is not a
multiple of `eval_freq`, everything driven by validation sees one event more, at an epoch it would
not otherwise see. Found independently by two reviewers:

```r
# eval_freq = 2, 3 epochs then 6, versus 6 in one go
uninterrupted history epochs: 2 4 6
resumed       history epochs: 2 3 4 6

# lr_reduce_on_plateau, patience 1, factor 0.5
uninterrupted batch lrs: 0.1 (18 values, never reduced)
resumed       batch lrs: 0.1 x12 then 0.05 x6      -> weights differ

# early stopping, patience 2
uninterrupted: 6 epochs        resumed: 4 epochs   -> weights differ
```

It does *not* occur in the kill-and-restart pattern, where the writing run is configured for the
same `epochs` and never reaches its own last epoch — only when the split is planned. Checkpointing
at multiples of `eval_freq` avoids it. A real fix would record in the checkpoint that its epoch was
a forced evaluation and not re-count it.

## R3-2. MEDIUM — a `torch` tensor in a callback state is destroyed by the checkpoint

`state<n>.rds` is written with `saveRDS`, which does not preserve torch tensors: they are external
pointers. Nothing warns at write time, the state file looks intact, and the *resume* dies with a
bare `external pointer is not valid` naming neither the callback nor the checkpoint. The same state
also breaks `$marshal()`, which serializes `network`/`optimizer`/`loss_fn` but not
`model$callbacks`.

```r
tcb = torch_callback("tcb",
  initialize = function() self$t = torch_tensor(1),
  on_epoch_end = function() self$t = self$t + 1,
  state_dict = function() list(t = self$t),
  load_state_dict = function(state_dict) self$t = state_dict$t)
# write 2 epochs with a checkpoint, then resume to 4:
# INFO Resuming training from the checkpoint in '...', which is at epoch 2.
# Error: external pointer is not valid
```

**Documented** — `?CallbackSet`'s *State* section, the `state_dict` argument of `callback_set()` /
`torch_callback()`, and the checkpoint's `state<n>.rds` bullet now say a state must survive
`saveRDS()` and point at `as.array()` / `torch::torch_serialize()`. **Open:** nothing enforces it.
Checking `states` for external pointers in `.save()` and erroring there would turn a resume-time
crash into a write-time message naming the callback.

## R3-3. LOW — whether an `on_end` state change reaches the checkpoint depends on `freq`

The callback has two write points: `on_epoch_end` when `freq` divides the epoch, and `on_end`
otherwise. `.save()` snapshots the other callbacks' states at whichever moment it runs, and the
checkpoint's `Inf` weight puts it last within its stage. So a state change a callback makes in
`on_end` is captured when `freq` did *not* already write that epoch, and lost when it did — for the
same callback and the same epochs.

```
# callback appends E<epoch> in on_epoch_end and END@<epoch> in on_end
uninterrupted 4 : E1,E2,E3,E4,END@4
resumed 2+2     : E1,E2,END@2,E3,E4,END@4     # freq = 5
resumed 2+2     : E1,E2,E3,E4,END@4           # freq = 1, by accident
```

No shipped callback is affected. **Fix:** collapse the two write points into one — `on_epoch_end`
knows whether it is the last epoch (`ctx$epoch == ctx$total_epochs`, or `ctx$terminate`, which
early stopping sets in the earlier `on_valid_end`), so `on_end` can go away entirely and a
checkpoint always means "the state as of the end of epoch n".

## R3-4. LOW — a resumed `history` loses its `epoch` key

`CallbackSetHistory$state_dict()` builds an uninterrupted history with `merge(train, valid, by =
"epoch")`, which returns a keyed `data.table`; the resumed branch returns `rbind(prev_state, state,
fill = TRUE)`, which is unkeyed. Same values, different object: `all.equal()` reports *"Datasets
have different keys"*, and keyed subsetting (`history[.(3)]`) works on one and errors on the other.
Only when both train and valid measures are set. **Fix:** `setkeyv(., "epoch")` on the `rbind`
branch, or drop the key in both.

## R3-5. LOW — `ctx$step` is not restored, unlike the other counters

`ctx$epoch`, `ctx$global_step`, `ctx$total_epochs` and `ctx$last_scores_valid` are all correct
across a resume, but `ctx$step` is `NULL` at the resumed run's first `on_epoch_begin`, where an
uninterrupted run shows the previous epoch's value (`s=NA` vs `s=3`). In a finished re-run (R2-7)
`ctx$step` and `ctx$last_scores_train` are `NULL` at `on_end` while `last_scores_valid` is restored.
**Fix:** set `ctx$step` in `resume_training()` and document which `ctx` fields a resumed run
restores.

## R3-6. LOW — a folder holding another callback's file gives an opaque checkmate error

A sibling callback that logs into the checkpoint folder makes it non-empty without making it a
checkpoint folder, so `can_checkpoint_into()` is `FALSE` and `initialize()` falls through to
`assert_path_for_output()`: `Assertion on 'path' failed: File at path already exists: '/tmp/...'`.
The user set `resume = TRUE` and gets a message that mentions neither checkpoints nor resuming.
**Fix:** replace the fallback with a `stopf()` in the same voice as the other checkpoint errors.

## R3-7. LOW — an error inside a user `load_state_dict()` says nothing about the callback

Restoring a state that a user's `load_state_dict()` rejects propagates the raw error (`boom in
load_state_dict`) with no mention of the callback id or that it happened while restoring a
checkpoint. **Fix:** wrap the `cb$load_state_dict(state)` call in `load_callback_states()` and
prefix the id and the folder.

## R3-8. LOW — `on_begin`/`on_end` run once per run, so state accumulated there grows

Stage counts across a 2+2 resume match an uninterrupted 4-epoch run exactly — `batch_end` 12,
`epoch_begin` 4, `epoch_end` 4, `end` 1 — except `begin`, which is 2. A callback that accumulates
in `on_begin` or `on_end` therefore gains an entry per restart, and a finished re-run (R2-7) still
calls `on_begin`/`on_end`/`on_exit` once more. Separately, and not resume-specific: `on_exit`
mutations never reach `learner$model$callbacks`, because the states are collected before the
`on.exit` handler fires. **Fix:** document both.

## R2-13. LOW — an architecture change with matching shapes resumes silently

`activation` (`nn_relu` → `nn_tanh`) and dropout `p` (0.1 → 0.9) changed between the runs are
accepted without a word: a different network is trained from the checkpoint's weights. A changed
`neurons` does error, but with `The size of tensor a (8) must match ...` after a 60-frame C++
backtrace, mentioning neither checkpoints nor the folder.

**Fix:** record the architecture-shaping `param_vals` in `state<n>.rds` and warn on a mismatch; wrap
`load_state_dict()` so the error names the checkpoint. (An mlr3-`HotstartStack`-style check that
every parameter but `epochs` is unchanged would subsume this; note that a `task$hash` cannot serve
as the task half of such a check, because `DataDescriptor` hashes `address(dataset)` and so a
lazy-tensor task's hash is not stable across constructions — hence the `run.rds` fingerprint used
for R2-1.)

## R2-14. LOW — the checkpoint folder's writability is never checked

`initialize()` accepts any existing empty or checkpoint-holding directory without probing
writability, so a read-only folder fails at the first `on_epoch_end` with `cannot open the
connection`. On iris that costs 0.1 s; on the multi-hour epochs this feature exists for it costs an
epoch — and it happens on the resume path too, for a folder that became read-only between runs.

**Fix:** `file.access(path, 2)` in `initialize()`, erroring with the folder name.

---

## Confirmed sound

**The core resume path is bit-exact.** With `shuffle = FALSE`, `p = 0` and a fixed `seed`, training
4 epochs in one go and training 2 + 2 across a resume produce **bitwise identical** networks —
plain, and with `lr_step`, `lr_cosine_annealing`, `lr_multiplicative`, `lr_reduce_on_plateau`,
`t_clbk("unfreeze")` (epoch-wise) and `jit_trace = TRUE`.

**No epoch or `global_step` off-by-one exists**, and this was specifically looked for: `ctx$epoch`
is restored to the checkpoint's epoch before `on_begin`, so `.start_epoch`, `freq`, `eval_freq`, the
`on_end` "already saved / trained nothing" guard and the batch-wise unfreeze schedule all line up
with an uninterrupted run. `resume_training()` running before the `begin` stage is correct and
load-bearing (the LR scheduler and unfreeze both depend on it).

**The restart loop itself works.** 14 real SIGKILL restarts of one script (`epochs = 12`,
`freq = 1`, `resume = TRUE`, a network slow enough to be interrupted mid-write) reached epoch 12
with no lost or duplicated epochs. Kills mid-epoch and between epochs, `epochs` raised between
attempts, `freq` changed between restarts, `freq > epochs`, `epochs = 0`, trailing-slash / relative
/ `~` paths, a deleted folder, and bad `resume` targets (a regular file, a missing folder, `""`)
all behave as intended.

**Damaged-folder handling**, apart from R2-8: deleting files of the *newest* checkpoint is accepted
and the epoch rewritten; deleting a file of an *older* one warns and resumes from the newest
complete; a foreign version string warns and continues.

**Checkpoint guards:** two runs cannot silently share a folder sequentially, `freq > 1` and early
termination write consistent folders, and `resample()`/`benchmark()` with a `path` function gives
one folder per fit (verified with 3 folds and 3 distinct folders, the function evaluated exactly
once per fit); a fixed `path` under `resample()` errors on the second fold instead of clobbering the
first.

**Callback state contract:** the unknown-id warning, the "callback does not implement
`$load_state_dict()`" warning and `assert_checkpoint_writes_last()` all behave as documented for the
shipped callbacks. Early stopping's `stagnation` / `best_score` / `best_epochs` carry across a
resume correctly.

**mlr3 integration:** sequential `resample()` with a `path` function gives one folder per fit and no
double evaluation; encapsulation with a fallback turns a checkpoint error into one learner error,
predicts from the fallback, and leaves the folder byte-identical; `$marshal()`/`$unmarshal()` and
`saveRDS()`/`readRDS()` around a resumed learner preserve `$model$epochs`,
`$internal_tuned_values` and predictions exactly.

**Config changes that behave:** an unchanged `batch_size` continues a batch-wise unfreeze exactly
like an uninterrupted run; `jit_trace` and `tensor_dataset` toggled between runs resume cleanly; a
changed `neurons` errors rather than loading a partial state.

**Round 3, with every parameter but `epochs` held equal**, comparing against an uninterrupted run
(`seed = 1`, `shuffle = FALSE`, `p = 0`) on full LR trajectories, all parameter tensors, callback
states and predictions:

* Every LR scheduler, epoch-wise and per-batch (`step_on_epoch = FALSE`): `lr_step`,
  `lr_multiplicative`, `lr_cosine_annealing`, `lr_lambda`, `lr_reduce_on_plateau` — bit-identical.
  `lr_one_cycle` on a genuinely killed run (subprocess killed after epoch 3, both runs configured
  for 6, SGD with momentum) is identical in learning rate *and* cycled momentum, which is the R1-2
  fix holding under the case that produced it; the planned-split case errors up front as documented.
* `t_clbk("unfreeze")`, epoch and batch schedules, with points before, exactly on and after the
  resume boundary: the per-batch set of trainable parameters is character-identical.
* Early stopping that has not yet fired: counters carry over, it fires at the same epoch, and
  `internal_tuned_values$epochs` / `internal_valid_scores` match. `restore_best_weights` warns and
  behaves exactly as documented.
* The whole stack at once (history, progress, an LR scheduler, unfreeze, early stopping,
  checkpoint) at `eval_freq = 1`: bit-identical, including chained resumes 2→4→6 and 1→2→…→6.
* Ordering: user callbacks at weights 0 / 500 / 2000 fire in the same relative order after a resume,
  a stateful callback between early stopping (1000) and the checkpoint (`Inf`) is captured and
  restored, and one at `Inf` after the checkpoint is refused with the documented error.
* `freq` not dividing `epochs` (`freq = 4`, `epochs = 7`, resume at 5): correct checkpoints, no
  clobbering, identical weights.
* Custom callbacks: scalars, lists and `data.table`s round-trip exactly; `load_state_dict()` runs
  before the resuming run's `on_begin`; stage counts match an uninterrupted run; `torch_callback()`
  refuses `state_dict` without `load_state_dict`, and a reserved id (`early_stopping`) is refused.
* Workflow: `resume = TRUE`; `saveRDS`/`readRDS` and `$marshal()`/`$unmarshal()` of a resumed
  learner keep the custom callback state and `model$callbacks$checkpoint$path`; a `resample()` fit's
  folder is readable off its learner and resumable; a finished re-run (R2-7) with the full stack
  writes nothing and returns identical weights, history, scores and predictions.

**Documentation:** every other claim checked came out true, including the epoch arithmetic, the
"never writes over an existing checkpoint" guarantee, `resume = TRUE` with and without a checkpoint
callback, the history's `NA` filling and its `eval_freq` behaviour, the progress split-time output,
the LR-scheduler continuation and the one-cycle total-steps error, and the `Inf` ordering weight.
`roxygenise()` + `tools::checkRd()` are clean on all 15 pages; `man/` is not stale.

## Coverage gaps

`AutoTuner`/`mlr3tuning` end to end with internal `epochs` tuning plus checkpointing; `benchmark()`;
resuming from a `saveRDS`-ed learner's folder in a genuinely fresh session; what a resume *produces*
from a folder corrupted by R2-2; `sampler`/`batch_sampler` changes; a per-batch LR scheduler
with a changed `batch_size`; `device` changes; `loss` and `loss.*` changes; two callbacks of the
same class; a custom callback with its own state.
Anything requiring `tfevents` is untestable here — the whole TensorBoard file-content story
("the curves extend those of the run it continues") remains unverified.

Round 3 additionally did not cover: `t_clbk("tb")` (no `tfevents` in this environment), custom
schedulers built with `torch::lr_scheduler()` that carry extra state, optimizers with more than one
param group, GPU and multi-worker dataloaders, and — by construction, since it held everything but
`epochs` fixed — the documented override behaviour for `opt.*` and scheduler arguments.

One environment caveat: `pkgload::load_all()` under `multisession` is itself flaky (workers
intermittently lose R6 methods), which limited how much parallel testing was possible.

---

## Already addressed

| # | Severity | Finding | Status |
|---|---|---|---|
| R1-1 | high | `CallbackSetHistory$state_dict()` binds without `fill`, killing a resumed run with `eval_freq > 1` and leaving a half-written checkpoint | **fixed** — `rbind(self$prev_state, state, fill = TRUE)`; regression test with `eval_freq = 2` |
| R1-2 | medium | `lr_one_cycle` resumes with the schedule's initial momentum instead of the annealed one (only `lr` was snapshotted before constructing the scheduler) | **fixed** — `on_begin()` snapshots the whole parameter groups, not just `lr`, and `.restore_scheduler_state()` re-applies all of them; test compares against an uninterrupted run |
| R1-3 | medium | A folder whose newest checkpoint is half-written is refused by `can_checkpoint_into()` while the reading side handles it fine, bricking the folder after a kill mid-write | **fixed** — such a folder is accepted and the partial epoch written over; only the checkpoint actually resumed from must be complete |
| R1-4 | medium | Resuming discards the resuming run's `opt.*` values in favour of the checkpoint's optimizer state (`load_state_dict()` replaces whole `param_groups`) | **documented as intended** — *Resuming* now says to configure the resuming learner like the one that wrote the checkpoint and change only `epochs`. R2-5 and R2-13 are cases where that contract does not hold or is not stated |
| R1-5 | low | `CallbackSetTB` claimed to be trivially resumable but its `assert_path_for_output()` refused its own log directory | **fixed** — an empty folder or one holding `events.out.tfevents.*` is accepted; the loss is logged against `ctx$global_step`; the file-content half remains unverified without `tfevents` |
| R1-6 | low | A `resume` path holding no checkpoint silently trained from scratch | **fixed** — now an error, except for an empty folder or one holding only incomplete checkpoints, so a restartable script still works |
| R2-1 | high | `validate = <ratio>` silently redraws the validation split on resume: the ratio split is drawn from R's RNG at train time and `seed` only seeds torch, so the resumed run trains on rows that were held out, validates on rows it has fitted, and compares early stopping's restored `best_score` against a different row set | **fixed** — the first checkpoint writes `run.rds` once per folder with the training task's id and the validation row ids; `assert_resumable_task()` errors before anything is loaded on a different task id, on a validation split appearing or disappearing, and on differing validation rows, naming `validate = "predefined"` and seeding R's RNG as the fixes. Documented in `LearnerTorch`'s *Resuming* section |
| R2-3 | medium | The `classes` check does not catch the case its documentation describes: `torch_callback()` derives the class name from the id, so a custom callback under a builtin's id has the builtin's class name and passes | **documented as a limitation** — the check stays a class-name comparison (it catches ids that stand for a callback of another class); the checkpoint docs now state that callbacks of the same class are indistinguishable to it and that a custom `torch_callback()` under a builtin's id is therefore not caught |
| R2-4 | medium | Changing or reordering `measures_valid` silently corrupted early stopping: only `measures_valid[[1L]]` is tracked and nothing recorded *which* measure `best_score` belonged to, so the resumed run compared the new measure's values against the old measure's score, using the new measure's `minimize` direction | **fixed** — the early stopping state carries the id of the measure its `best_score` belongs to, and `load_state_dict()` errors when this run's first validation measure is a different one. Test *"refuses a checkpoint whose best score belongs to another validation measure"* covers a changed measure, a merely reordered one, and the case that a further measure is appended |
| R2-5 | medium | A fixed `seed` makes every resumed run replay the same batch order: torch's generator is seeded at the start of every `$train()` and no RNG state is checkpointed, so the resumed epochs repeat the shuffling and dropout masks of the run's first epochs | **documented** — `LearnerTorch`'s *Resuming* section now spells the replay out next to "rng states are not restored", and points at the default `seed = "random"` for runs where this matters more than reproducing a run exactly. Checkpointing `torch_get_rng_state()` would fix it properly |
| R2-6 | medium | Changing `batch_size` skipped a batch-wise `unfreeze` schedule, because `on_batch_begin()` re-derived `(epoch - 1) * length(loader_train) + step` instead of reading the restored counter | **fixed** — the callback reads `self$ctx$global_step`, which is checkpointed and counted up per batch, so a schedule point is never jumped over |
| R2-7 | medium | A finished run left the restart script permanently erroring: a kill after the last epoch but before the script's own follow-up work was unrecoverable, since `resume` refused a checkpoint that was already at `epochs` and nothing else turns a checkpoint folder into a trained learner | **fixed** — only `epochs` *less* than the checkpoint is an error now; at exactly `epochs` the run in the folder is finished, so it is restored, no epoch is trained and its model is returned. The checkpoint callback writes nothing (`on_begin` lets a run that trains nothing past the "would not get past it" guard, and `on_end` already returned early), and the last validation scores are now part of the state file so such a run still reports `internal_valid_scores` |
| R2-10 | medium | The folder a `path` function chose was unrecoverable from the learner: the callback held no state, so nothing about the evaluated path reached `learner$model` and `param_set$values` still held the function — under the documented `resample()` idiom no fold could be resumed | **fixed** — the callback's `state_dict()` returns its resolved `path`, which lands in `learner$model$callbacks$<id>$path`. It is deliberately kept out of `state<n>.rds` and never restored: a resuming run writes where its own `path` says. Covered by the `resample()` test, which now checks the three fits' folders against the learners |
| R2-12 | high | A run that early stopping ended resumed and un-stopped itself: `ctx$terminate` was consulted only *after* an epoch, so a restart always trained one more — and if that epoch improved, `stagnation` reset and the run continued to its full budget (4 epochs and `internal_tuned_values$epochs` 2 became 20 and 20) | **fixed** — `CallbackSetEarlyStopping$load_state_dict()` sets `ctx$terminate` when the restored `stagnation` has reached `patience`, and the loop now tests `terminate` before each epoch instead of after it. Such a run warns, trains nothing and returns the checkpoint's model. The same contract is documented for user callbacks that terminate: re-establish the flag in `$load_state_dict()`. The test that hid this (`opt.lr = 0`, so no epoch can improve) was replaced by one where the extra epoch could improve |
| R2-15 | low | `CallbackSetUnfreeze` has state but no *Resuming* section, while `LearnerTorch` promises that the callbacks document their behaviour under one | **fixed** — its page now has one: what is restored, that both an `epoch` and a `batch` schedule continue rather than start over, and what a changed `batch_size` does to a `batch` schedule. `CallbackSetEarlyStopping` has no page of its own (it exists only through `patience`), so its resume behaviour stays in `LearnerTorch`'s *Resuming* section, which now also covers the validation-measure check |

Two round-1 suspicions were settled: a `CallbackSetProgress` state without `elapsed` is not
reachable with the current package, and no epoch or `global_step` off-by-one exists. The other two —
a fixed `path` under parallelisation, and nothing in a checkpoint identifying the task — became
R2-2 and R2-1 (task identity is now covered; the learner/architecture half is R2-13).
