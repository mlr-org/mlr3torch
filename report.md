# Bug review: checkpoint / resume (`feat/checkpoint-resume`)

Scope: `git diff main...HEAD` plus the uncommitted working-tree changes, in the worktree
`/Users/sebi/mlr/mlr3torch/.claude/worktrees/resume-stack`. Findings are ordered by severity.

Note on naming: while this review was running, the learner parameter was renamed from `path` to
`resume`. The findings below are independent of that rename; repros use `resume`.

Correctness of the core resume path was checked end-to-end and is sound: with `shuffle = FALSE`,
`p = 0` and a fixed `seed`, training 4 epochs in one go and training 2 + 2 epochs across a resume
produce **bitwise identical** networks -- plain, and with `lr_step`, `lr_cosine_annealing`,
`lr_multiplicative`, `lr_reduce_on_plateau`, `t_clbk("unfreeze")` (epoch-wise), and `jit_trace =
TRUE`. Early stopping's `stagnation` / `best_score` / `best_epochs` carry across a resume correctly,
`resume = TRUE` works, the `classes` mismatch check and the unknown-id warning fire as documented,
a `path` function is evaluated exactly once per fit (verified with `resample()`, 3 folds, 3
distinct folders), and a fixed `path` under `resample()` errors on the second fold instead of
clobbering the first. The bugs below are the exceptions.

---

## 1. HIGH -- `CallbackSetHistory$state_dict()` kills a resumed run and bricks the checkpoint folder

**File / function:** `R/CallbackSetHistory.R`, `state_dict()` -- the final
`rbind(self$prev_state, state)` -- interacting with `CallbackSetCheckpoint$.save()`.

**What goes wrong.** `state` is `data.table(epoch = numeric(0))` -- a *single* column -- in any epoch
in which neither `ctx$last_scores_train` nor `ctx$last_scores_valid` was produced. `prev_state` is
non-`NULL` only on a resumed run and carries the columns of the previous run. `rbind()` is called
without `fill = TRUE`, so binding a 1-column table onto a 3-column one throws

```
rbindlist(): Item 2 has 1 columns, inconsistent with item 1 which has 3 columns.
```

The checkpoint callback calls `$state_dict()` on **every** epoch it writes, so this fires in the
first epoch of the resumed run that does not evaluate. This is resume-only: on a fresh run
`prev_state` is `NULL` and `state` is returned untouched.

**Trigger (verified by running).** `tsk("iris")`, `t_clbk("history")` +
`t_clbk("checkpoint", freq = 1)`, `eval_freq = 2`, `measures_train` and `measures_valid` set;
train with `epochs = 2`, then resume with `epochs = 6`. The resumed run dies at the end of epoch 3.
The same happens whenever the measure set differs between the checkpointing and the resuming run
(e.g. run 1 with measures, run 2 without), in *every* epoch.

**Why it is worse than a plain error.** The exception is raised inside `.save()` *after*
`network3.pt` and `optimizer3.pt` have been written but *before* `state3.rds`. The folder is left
with an incomplete newest checkpoint, and `can_checkpoint_into()` then refuses it permanently (see
finding 3). Retrying the same command gives

```
Assertion on 'path' failed: File at path already exists: '/tmp/.../fileXXXX'.
```

so a long training run is lost and cannot be restarted into its own folder without hand-deleting
files. Verified end to end.

**Suggested fix.** `rbind(self$prev_state, state, fill = TRUE)`, and/or return `self$prev_state`
unchanged when `state` has no rows. Worth a test with `eval_freq > 1` + `freq = 1` + `history` +
resume -- the branch's history tests all use `eval_freq = 1`, which is why this was not caught.

---

## 2. MEDIUM -- `lr_one_cycle` resumes with the wrong momentum for the first batch

**File / function:** `R/CallbackSetLRScheduler.R`, `private$.restore_scheduler_state()` (and the
`lrs = map_dbl(...$param_groups, function(group) group$lr)` snapshot in `on_begin()`).

**What goes wrong.** The code correctly recognises that constructing a scheduler resets the
optimizer's learning rate to the schedule's *initial* rate, and therefore snapshots `group$lr`
before construction and writes it back after `load_state_dict()`. But it snapshots **only `lr`**.
`torch::lr_one_cycle()` with its default `cycle_momentum = TRUE` also drives the optimizer's
momentum (`betas[[1]]` for Adam, `momentum` for SGD), and its constructor likewise resets that to
the start-of-cycle value. Nothing puts it back, so the first `optimizer$step()` of the resumed run
runs with the momentum of step 0 instead of the momentum of the step it resumed at. The scheduler
recomputes momentum from `last_epoch` on its next `step()`, so the damage is confined to one batch
-- but it is silent and it propagates through Adam's moment buffers.

**Verified by running.** iris, `lr_one_cycle(max_lr = 0.1)`, batch size 50 (3 batches/epoch),
4 epochs; a crash callback aborts the first run at epoch 3, the second run resumes to epoch 4:

```
FULL RUN                            RESUMED RUN
  gs=7 lr=0.064738 beta1=0.885262     gs=7 lr=0.064738 beta1=0.950000   <-- wrong
  gs=8 lr=0.046264 beta1=0.903737     gs=8 lr=0.046264 beta1=0.903737
```

Max absolute difference in the first weight matrix at the end of training: `4.32e-2`. With
`cycle_momentum = FALSE` the difference is exactly `0`, which isolates momentum as the cause.

**Suggested fix.** Snapshot and restore every scalar the scheduler may write into the param groups,
not just `lr` -- at minimum `momentum` and `betas` -- e.g. keep a copy of the relevant param-group
entries before constructing the scheduler and re-apply all of them in `.restore_scheduler_state()`.

---

## 3. MEDIUM -- the writing side refuses a folder the reading side handles fine

**File / functions:** `R/CallbackSetCheckpoint.R` -- `can_checkpoint_into()` and
`CallbackSetCheckpoint$initialize()` versus `checkpoint_suffixes()` / `latest_checkpoint()`.

**What goes wrong.** The comment on `checkpoint_files()` states that reading and writing "go through
this, so they cannot disagree on what 'exists' means". They do disagree:

* Reading: `latest_checkpoint()` -> `checkpoint_suffixes()` **warns about and ignores** incomplete
  checkpoints and resumes from the newest *complete* one.
* Writing: `can_checkpoint_into()` returns `FALSE` whenever `max(incomplete) > max(complete)`, and
  `initialize()` then falls through to `assert_path_for_output(path)`, which fails with the generic
  `File at path already exists: '<dir>'`.

The docblock justifies the refusal with "that is the checkpoint a resuming run would continue from",
but that is not what the resuming run does -- it continues from the newest *complete* checkpoint,
and `on_begin()`'s clash check only ever protects *complete* checkpoints, so rewriting the partial
epoch would be safe.

**Consequence.** The exact failure the incomplete-detection exists for -- a process killed while
writing a checkpoint -- makes the folder permanently unusable for any run that also has a checkpoint
callback, i.e. the normal crash-recovery configuration. The error message says nothing about
checkpoints and gives no recovery hint. This is also what turns finding 1 from "one failed run"
into "the folder is dead".

**Verified by running.** With `network3.pt` present but `optimizer3.pt` / `state3.rds` deleted:
a learner with `resume = <folder>` and **no** checkpoint callback warns
(`Ignoring incomplete checkpoint(s) 3 in '...'`) and resumes from epoch 2 without trouble; the same
learner *with* `t_clbk("checkpoint")` on the same folder dies in the callback constructor with
`File at path already exists`.

**Suggested fix.** Allow the folder when the incomplete epochs are ones this run is going to write
anyway (the `on_begin()` clash check already guards the real danger), or -- if the refusal is
intentional -- raise a dedicated error from `initialize()` that names the partial files and says to
delete them, rather than letting `assert_path_for_output()` produce the message.

---

## 4. MEDIUM -- resuming silently ignores the optimizer hyper-parameters the resuming run configures

**File / function:** `R/learner_torch_methods.R`, `resume_training()`:
`ctx$optimizer$load_state_dict(torch_load(checkpoint$optimizer))`.

**What goes wrong.** `torch`'s `load_state_dict()` replaces the whole `param_groups`, including
`lr`, `weight_decay`, `betas`, ... . So a resuming learner configured with a different `opt.lr` (or
any other optimizer parameter) trains with the checkpoint's value, with no warning anywhere.

**Verified by running.** Run 1 with `opt.lr = 0.1` for 2 epochs; resumed run configured with
`opt.lr = 1e-6` for 4 epochs -> `learner$model$optimizer$param_groups[[1]]$lr` is `0.1`, and the
epochs were trained at `0.1`.

"Resume the run with a lower learning rate" is one of the main reasons people resume, so this
silently produces a model different from the one the user configured. The *Resuming* documentation
only says "optimizer states" are loaded; nothing warns that the configured hyper-parameters are
discarded. (Changing the optimizer *class* does error, loudly, from inside `torch`.)

**Suggested fix.** After `load_state_dict()`, either re-apply the configured `opt.*` values onto the
restored param groups, or compare them against the restored ones and warn/error on a difference --
the same treatment `CallbackSetLRScheduler` already documents for the schedule's arguments.

---

## 5. LOW -- `CallbackSetTB`'s new *Resuming* section contradicts its constructor

**File:** `R/CallbackSetTB.R`.

The added section says "This callback keeps no state of its own so it can trivially be resumed", but
`initialize()` calls `assert_path_for_output(path)` unconditionally, so pointing a resumed run at
the log directory the first run wrote errors with `File at path already exists`. **Verified by
running** `CallbackSetTB$new(p, FALSE)` twice on the same path. A resumed run therefore has to be
given a *fresh* log dir, which splits one logical training run across two TensorBoard directories --
the opposite of "trivially resumed".

Secondary, from the same diff: epoch-level scores are logged with `step = ctx$epoch` while
`.log_train_loss()` now logs with `step = ctx$global_step`. Two different step scales in one log
directory make the TensorBoard x-axis meaningless when both are enabled. (Read only.)

**Suggested fix.** Either allow an existing directory in `CallbackSetTB$initialize()` (so a resumed
run appends to the same run) and adjust the doc, or drop the "trivially resumed" claim and say a
fresh path is required. Pick one step scale, or log the loss against a fractional epoch.

---

## 6. LOW -- a mistyped `resume` path silently trains from scratch

**File / function:** `R/learner_torch_methods.R`, `resume_training()`.

When `latest_checkpoint()` returns `NULL` the run logs at `info` level and trains from scratch. This
is deliberate ("so that the same script can be used for the first run and for restarts"), and when
the checkpoint callback writes into the same folder a typo is caught later by `on_begin()`'s clash
check. But a learner that resumes *without* a checkpoint callback -- a perfectly reasonable "continue
this run for a few more epochs" setup -- gets a from-scratch model with nothing above `info` level to
say so. **Verified by running:** `resume = tempfile()` (non-existent), `epochs = 2` -> trains 2 epochs
from random initialisation, no warning.

**Suggested fix.** Distinguish "the folder does not exist / is not empty but holds no checkpoints"
(warn) from "the folder exists and is empty" (the intended first-run case, info is fine).

---

## Category summary

* **Correctness of resuming (category 1):** one high-severity bug (finding 1). Off-by-one in epochs
  and `global_step` was specifically looked for and is **not** present: `ctx$epoch` is restored to
  the checkpoint's epoch before `on_begin`, so `.start_epoch`, `freq`, `eval_freq`, the `on_end`
  "already saved / trained nothing" guard and the batch-wise unfreeze schedule all line up with an
  uninterrupted run. `resume_training()` running before the `begin` stage is correct and load-bearing
  (the LR scheduler and unfreeze both depend on it).
* **Checkpoint guards (category 2):** finding 3, plus finding 1's folder-bricking side effect. The
  guards themselves are effective: two runs cannot silently share a folder, `freq > 1` and early
  termination write consistent folders, and `resample()`/`benchmark()` with a `path` function gives
  one folder per fit.
* **Callback state contract (category 3):** no bug found beyond finding 2. The `classes` check, the
  unknown-id warning and the "callback does not implement `$load_state_dict()`" warning all behave
  as documented, and `assert_checkpoint_writes_last()` is sound for the shipped callbacks.
* **`restore_best_weights` after a resume** is a real silent-wrong-result but it is already known,
  documented in `man-roxygen/paramset_torchlearner.R`, warned about in
  `CallbackSetEarlyStopping$load_state_dict()`, and written up in `PROBLEM.md`; not re-reported.

---

## Suspicions I could not confirm

* **Parallel `resample()`/`benchmark()` with a fixed `path`.** The `can_checkpoint_into()` /
  `on_begin()` guards are checked per worker, so two workers that both see an empty folder would
  both write `network1.pt` and the last writer wins. I did not reproduce this (it needs a real
  parallel backend and a race). Sequentially the second fold errors correctly. The `path`-as-function
  recommendation already covers this; a note in the docs that a *fixed* path is unsafe under
  parallelisation would be cheap.
* **Nothing in a checkpoint identifies the task or the learner.** Resuming a checkpoint that was
  written for a different task with the same feature/class shapes succeeds silently (verified that
  no check exists; a genuinely different shape errors loudly from `torch`). This looks inherent to
  the feature rather than a defect in the branch, so I did not rank it.
* **`CallbackSetProgress$load_state_dict()`** sets `private$.elapsed = state_dict$elapsed`. For a
  state written by a version whose progress state lacked `elapsed`, `.elapsed` becomes `NULL`,
  `.total_elapsed()` returns `numeric(0)` and `on_end()`'s `if (private$.elapsed == 0)` would fail
  with "argument is of length zero". Only reachable across versions, so I could not construct it
  with the current package; a `%??% 0` would remove the risk.
* **`train_loop()` reads the resume path from `ctx$learner$param_set$values$resume` while
  `learner_torch_train()` validates `param_vals$resume`.** For every path I checked these agree
  (`$get_values(tags = "train")` and `$values` return the same thing for this parameter), so I could
  not turn the inconsistency into a bug -- but taking it from `param_vals`, like every other
  parameter, would close the question.
