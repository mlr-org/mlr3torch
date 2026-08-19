# Resume review, round 2 — four user perspectives

Four reviewers went at `feat/checkpoint-resume` in parallel, each emulating a kind of user
rather than reading the diff: an **mlr3 power user** (resample/benchmark/tuning/parallel/marshal),
a **cluster operator** (the kill-and-restart loop), a **config-fiddler** (change one setting
between the two runs), and a **first-time reader** (check every documented claim by running it).
Everything below was verified by running code; each reviewer was timeboxed, so the coverage gaps
at the end are real.

The findings of the earlier `report.md` are not repeated here; their current status is in
*Already addressed* at the bottom.

| # | Severity | Finding | Area |
|---|---|---|---|
| 1 | **high** | `validate = <ratio>` silently redraws the validation split on resume | correctness |
| 2 | med-high | Two concurrent runs share a checkpoint folder undetected | guards |
| 3 | medium | The `classes` check does not catch the case its documentation describes | guards |
| 4 | medium | Changing or reordering `measures_valid` silently corrupts early stopping | correctness |
| 5 | medium | A fixed `seed` makes every resumed run replay the same batch order | correctness |
| 6 | medium | Changing `batch_size` skips a batch-wise `unfreeze` schedule | correctness |
| 7 | medium | A finished run leaves the restart script permanently erroring | operability |
| 8 | medium | A *corrupted* newest checkpoint bricks the folder, with no fallback | operability |
| 9 | medium | `resume = A` while checkpointing into B works exactly once | operability |
| 10 | medium | The folder a `path` function chose is unrecoverable from the learner | ergonomics |
| 11 | medium | The documented per-fit-folder idiom breaks under `multisession` | ergonomics |
| 12 | low | A run that early-stopped trains further on every restart | correctness |
| 13 | low | An architecture change with matching shapes resumes silently | guards |
| 14 | low | The checkpoint folder's writability is never checked | operability |
| 15 | low | `CallbackSetUnfreeze` has state but no *Resuming* section | docs |

---

## 1. HIGH — `validate = <ratio>` silently redraws the validation split on resume

Found independently by two reviewers, with different repros. The ratio split is drawn from R's RNG
at train time and `seed` only seeds torch, so a resumed run gets a **different train/validation
partition** from the run it continues. Three consequences, none of them signalled:

* the resumed run trains on rows that were held out before;
* `$internal_valid_scores` is computed largely on rows the model has already fitted (optimistic);
* early stopping's restored `best_score` is compared against scores from a different row set.

End to end: run 1 (`epochs = 3`, `patience = 3`) leaves `best_score = 0.2889`; the resumed run
validates on a different subset, never "improves", hits `stagnation = 3`, stops at epoch 4 and
reports `internal_tuned_values$epochs = 1`. Only 12 of 45 validation rows survived the resume.

```r
mk = function(ep, res = NULL) lrn("classif.mlp", epochs = ep, batch_size = 50, neurons = 5,
  seed = 1, validate = 0.3, measures_valid = msr("classif.acc"), resume = res,
  callbacks = t_clbk("checkpoint", path = p, freq = 1))
p = tempfile(); mk(2)$train(tsk("iris")); mk(4, res = p)$train(tsk("iris"))
# the two runs' task_valid$row_ids overlap in 12 of 45 rows
```

Unlike the `opt.*` case this cannot be avoided by configuring the resuming learner identically —
no setting reproduces the split. **Fix:** store the validation row ids in the checkpoint state and
restore them, or at minimum error when the resuming run's `task_valid$row_ids` differ from the
checkpointed ones. Document that `validate = "predefined"` / a fixed `internal_valid_task` is the
safe form across a resume.

## 2. MED-HIGH — two concurrent runs share a checkpoint folder undetected

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
complete. **Fix:** make folder ownership atomic (`dir.create()` returning `FALSE` is a reliable
"someone else has it", or an owner id in a `run.rds` re-checked in `on_begin()`), and re-verify the
clash set immediately before each `.save()` rather than only once up front.

## 3. MEDIUM — the `classes` check does not catch the case it documents

The docs say recording each callback's class "lets a resuming run notice that an id now stands for
a different callback". It does not, for the case that actually occurs: `torch_callback(id = ...)`
derives the class name from the id, so a custom callback given a builtin's id has *exactly* the
builtin's class name and passes the check. Verified: a run checkpointed with `t_clbk("history")`,
resumed with `torch_callback("history", state_dict = ..., load_state_dict = ...)`, hands the
builtin history's `data.table` straight to the custom `load_state_dict()` — no error, no warning.
The check only fires for builtin-vs-builtin, which cannot share an id anyway.

**Fix:** record something identity-bearing (the generator object, or the `TorchCallback`'s `man` /
package) rather than a class name derived from the id — or weaken the doc to say the check only
catches differing R6 class names, and that two `torch_callback()`s under one id are
indistinguishable.

## 4. MEDIUM — changing or reordering `measures_valid` corrupts early stopping

`CallbackSetEarlyStopping` restores `best_score`/`stagnation` but records nothing about *which*
measure produced them, and always tracks `measures_valid[[1L]]`. Resuming with a different or
merely reordered first measure compares the new measure's values against the old measure's score,
using the new measure's `minimize` direction — training then stops almost at once or never, and
`$model$callbacks$early_stopping$best_score` still holds the old measure's value.

```r
mk = function(ep, m, res = NULL) lrn("classif.mlp", epochs = ep, batch_size = 50, neurons = 5,
  seed = 1, validate = 0.3, measures_valid = m, patience = 2, resume = res,
  callbacks = t_clbk("checkpoint", path = p, freq = 1))
p = tempfile(); mk(2, msr("classif.acc"))$train(tsk("iris"))   # best_score = 0.333, an accuracy
mk(10, msr("classif.ce"), res = p)$train(tsk("iris"))$model$epochs   # 3 of 10, silently
```

**Fix:** record `measures_valid[[1L]]$id` and `$minimize` next to `best_score`, and error — or warn
and reset the early-stopping state — when they do not match on resume.

## 5. MEDIUM — a fixed `seed` makes every resumed run replay the same batch order

`torch_manual_seed()` runs at the start of every `$train()` and no RNG state is checkpointed, so
epochs 3–4 of a resume see byte-identical shuffling and dropout masks to epochs 1–2 of the original.
In the canonical use of this feature — checkpoint every epoch, restart on a preemptible machine —
the model sees the *same* batch order in every epoch. It hits exactly the user who follows the docs
and sets a seed for reproducibility; with the default `seed = "random"` it does not occur. The docs
say RNG is not restored, but not this.

**Fix:** checkpoint `torch_get_rng_state()` (and the CUDA state) and restore it; failing that,
derive the effective seed from the resumed epoch and document the replay.

## 6. MEDIUM — changing `batch_size` skips a batch-wise `unfreeze` schedule

`CallbackSetUnfreeze$on_batch_begin()` computes `(epoch - 1) * length(loader_train) + step` itself
instead of reading the restored `ctx$global_step`. A resume at a different `batch_size` — the
natural move when continuing on other hardware — renumbers the batches and jumps over the scheduled
unfreeze point entirely; the weights stay frozen and nothing is logged, because the existing
"No weights unfrozen at batch" warning only fires when the number *is* hit.

**Note:** this is a regression. The callback was changed to read `ctx$global_step` when the counter
moved into `ContextTorch`; the working tree has since reverted to the derived form. Re-applying that
one line fixes the finding.

## 7. MEDIUM — a finished run leaves the restart script permanently erroring

The restart idiom converges right up to the last epoch and then wedges: a kill *after* the final
epoch but before the script's own follow-up work (predict, `saveRDS`, upload) is unrecoverable. The
model exists only in the dead process, the checkpoint holds state dicts and no learner, and the
restart errors rather than finishing. The message advises raising `epochs`, which would train more
than asked and still not reproduce the requested model.

```r
mk = function() lrn("classif.mlp", epochs = 3, batch_size = 50, neurons = 5, resume = TRUE,
  callbacks = t_clbk("checkpoint", path = p, freq = 1))
p = tempfile(); mk()$train(tsk("iris"))   # ok
mk()$train(tsk("iris"))                   # ERROR: already trained for 3 epochs, but 'epochs' is 3
```

**Fix:** when the newest complete checkpoint is exactly `epochs`, treat the run as finished — load
it and return a trained model (the loop body never executes and `on_end` writes nothing, since
`epoch == .start_epoch`) instead of erroring. At minimum, say in the message that the run is
complete and how to get the model out.

## 8. MEDIUM — a corrupted newest checkpoint bricks the folder, with no fallback

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

## 9. MEDIUM — `resume = A` while checkpointing into B works exactly once

Keeping the original folder read-only and checkpointing the continuation elsewhere is a natural
operator move. On the second run `.start_epoch` comes from A, so every epoch this run plans
collides with what it wrote into B, and `on_begin()` refuses — at any `epochs`. The message
("written by another run … continue that run instead of starting it over") is wrong here: the
clashing checkpoints were written by this same script, and the fix is to point `resume` at B.

**Fix:** when the clashing epochs in the write folder continue the same lineage, say so —
"'<B>' already continues this run up to epoch 6; set `resume` to '<B>'".

## 10. MEDIUM — the folder a `path` function chose is unrecoverable from the learner

The checkpoint callback holds no state, so nothing about the evaluated path reaches
`learner$model` (`model$callbacks` is empty) and `param_set$values$cb.checkpoint.path` is still the
function. With the documented `resample()` idiom plus `store_models = TRUE` there is no way to learn
which folder a fold wrote to, so a specific fold cannot be resumed.

**Fix:** give the callback a `state_dict()` returning `list(path = self$path)` — it lands in
`model$callbacks` for free and makes `resume = TRUE` inspectable.

## 11. MEDIUM — the documented per-fit-folder idiom breaks under `multisession`

A `path` function written the obvious way keeps `globalenv()` as its enclosure, which is not
serialised to workers: `Error in path() : object 'i' not found`. Wrapping it in `local()` fixes
serialisation but gives each worker its own counter, so workers collide on folder names —
sequentially that errors, concurrently it is finding 2.

**Fix:** document that the function must be self-contained and must not depend on process-local
state; the safe recipes are `function() tempfile()` or a folder keyed on something globally unique.

## 12. LOW — a run that early-stopped trains further on every restart

Unlike a completed run (finding 7, which errors), an early-stopped run's restart succeeds and
trains past the point early stopping chose. The same script run four times gave `model$epochs` of
3, 4, 7, 10 and `internal_tuned_values$epochs` of 1, 1, 5, 8 — four different models.

**Fix:** record that the run terminated early and refuse or warn when resuming past it.

## 13. LOW — an architecture change with matching shapes resumes silently

`activation` (`nn_relu` → `nn_tanh`) and dropout `p` (0.1 → 0.9) changed between the runs are
accepted without a word: a different network is trained from the checkpoint's weights. A changed
`neurons` does error, but with `The size of tensor a (8) must match ...` after a 60-frame C++
backtrace, mentioning neither checkpoints nor the folder.

**Fix:** record the architecture-shaping `param_vals` in `state<n>.rds` and warn on a mismatch; wrap
`load_state_dict()` so the error names the checkpoint.

## 14. LOW — the checkpoint folder's writability is never checked

`initialize()` accepts any existing empty or checkpoint-holding directory without probing
writability, so a read-only folder fails at the first `on_epoch_end` with `cannot open the
connection`. On iris that costs 0.1 s; on the multi-hour epochs this feature exists for it costs an
epoch — and it happens on the resume path too, for a folder that became read-only between runs.

**Fix:** `file.access(path, 2)` in `initialize()`, erroring with the folder name.

## 15. LOW — `CallbackSetUnfreeze` has state but no *Resuming* section

`LearnerTorch`'s *Resuming* section promises "the callbacks document their behavior under a
corresponding *Resuming* section". `CallbackSetUnfreeze` implements `state_dict()`/
`load_state_dict()` and is restored on resume, but its page has no such section — so a user asking
what happens to an unfreeze schedule across a resume (finding 6, in particular) finds nothing.
`CallbackSetEarlyStopping`'s resume behaviour likewise lives only in the learner's
`restore_best_weights` bullet.

---

## Confirmed sound

**The restart loop itself works.** 14 real SIGKILL restarts of one script (`epochs = 12`,
`freq = 1`, `resume = TRUE`, a network slow enough to be interrupted mid-write) reached epoch 12
with no lost or duplicated epochs. Kills mid-epoch and between epochs, `epochs` raised between
attempts, `freq` changed between restarts, `freq > epochs`, `epochs = 0`, trailing-slash / relative
/ `~` paths, a deleted folder, and bad `resume` targets (a regular file, a missing folder, `""`)
all behave as intended.

**Damaged-folder handling**, apart from finding 8: deleting files of the *newest* checkpoint is
accepted and the epoch rewritten; deleting a file of an *older* one warns and resumes from the
newest complete; a foreign version string warns and continues.

**mlr3 integration:** sequential `resample()` with a `path` function gives one folder per fit and no
double evaluation; encapsulation with a fallback turns a checkpoint error into one learner error,
predicts from the fallback, and leaves the folder byte-identical; `$marshal()`/`$unmarshal()` and
`saveRDS()`/`readRDS()` around a resumed learner preserve `$model$epochs`,
`$internal_tuned_values` and predictions exactly.

**Config changes that behave:** an unchanged `batch_size` continues a batch-wise unfreeze exactly
like an uninterrupted run; `jit_trace` and `tensor_dataset` toggled between runs resume cleanly; a
changed `neurons` errors rather than loading a partial state.

**Documentation:** every other claim checked came out true, including the epoch arithmetic, the
"never writes over an existing checkpoint" guarantee, `resume = TRUE` with and without a checkpoint
callback, the unknown-id warning, the history's `NA` filling and its `eval_freq` behaviour, the
progress split-time output, the LR-scheduler continuation and the one-cycle total-steps error, and
the `Inf` ordering weight. `roxygenise()` + `tools::checkRd()` are clean on all 15 pages; `man/` is
not stale.

## Coverage gaps

`AutoTuner`/`mlr3tuning` end to end with internal `epochs` tuning plus checkpointing; `benchmark()`;
resuming from a `saveRDS`-ed learner's folder in a genuinely fresh session; what a resume *produces*
from a folder corrupted by finding 2; `sampler`/`batch_sampler` changes; a per-batch LR scheduler
with a changed `batch_size`; `device` changes; a different task with matching shapes; `loss` and
`loss.*` changes; two callbacks of the same class; a custom callback with its own state.
Anything requiring `tfevents` is untestable here — the whole TensorBoard file-content story
("the curves extend those of the run it continues") remains unverified.

One environment caveat: `pkgload::load_all()` under `multisession` is itself flaky (workers
intermittently lose R6 methods), which limited how much parallel testing was possible.

---

## Already addressed (round 1, from `report.md`)

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | high | `CallbackSetHistory$state_dict()` binds without `fill`, killing a resumed run with `eval_freq > 1` and leaving a half-written checkpoint | **fixed** — `rbind(..., fill = TRUE)`; regression test with `eval_freq = 2` |
| 2 | medium | `lr_one_cycle` resumes with the schedule's initial momentum instead of the annealed one | **fixed** — `on_begin()` snapshots the whole parameter groups, not just `lr`; test compares against an uninterrupted run |
| 3 | medium | A folder whose newest checkpoint is half-written is refused, bricking it after a kill mid-write | **fixed** — such a folder is accepted and the partial epoch written over; only the checkpoint actually resumed from must be complete |
| 4 | medium | Resuming discards the resuming run's `opt.*` values in favour of the checkpoint's optimizer state | **documented as intended** — *Resuming* now says to configure the resuming learner like the one that wrote the checkpoint and change only `epochs`. Findings 1, 4, 5, 6 and 13 above are cases where that contract does not hold or is not stated |
| 5 | low | `CallbackSetTB` claimed to be trivially resumable but refused its own log directory | **fixed** — an empty folder or one holding `events.out.tfevents.*` is accepted; the file-content half remains unverified without `tfevents` |
| 6 | low | A `resume` path holding no checkpoint silently trained from scratch | **fixed** — now an error, except for an empty folder or one holding only incomplete checkpoints, so a restartable script still works |

Two suspicions from that round were also settled: the `CallbackSetProgress` state without `elapsed`
is not reachable, and no epoch or `global_step` off-by-one exists. The remaining open ones — a
fixed `path` under parallelisation, and nothing in a checkpoint identifying the task — are now
findings 2 and 13 above.
