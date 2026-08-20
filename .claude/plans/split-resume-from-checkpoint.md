# Splitting `feat/resume-from-checkpoint`

Status: implemented as a stack of four branches on top of `main`.

## Where the work came from

`feat/resume-from-checkpoint` was 2 commits, ~1400 lines. The second commit redesigned the first:
commit 1 implemented resuming as a callback (`t_clbk("resume")`), commit 2 as a `path` parameter on
`LearnerTorch`, but never deleted commit 1's code, so both APIs shipped in parallel. The
`path`-parameter design is the one that was kept, per commit 2's rationale (matching Lightning's
`trainer.fit(ckpt_path=)` and Keras' `initial_epoch=`); `R/CallbackSetResume.R`, its `.Rd` and
`tests/testthat/test_CallbackSetResume.R` (~570 lines) were dropped.

`fix/checkpoint-files` (PR #481) is merged into `main` as `75a19f81`. It removed `freq_type`, which
means a checkpoint's file suffix is always its epoch -- the resume code no longer needs the
"correct as long as `freq_type` was epoch" caveat, and the state file records no `step`.

## The stack

Each branch is based on the previous one and is meant to be merged in this order.

| # | Branch | Content |
|---|--------|---------|
| 1 | `feat/ctx-callbacks` | `ContextTorch$callbacks`; `ctx$epoch = 0` before the `on_begin` stage |
| 2 | `feat/callback-state-dicts` | `$state_dict()` / `$load_state_dict()` for lr_scheduler, early_stopping, unfreeze |
| 3 | `feat/checkpoint-state` | `state<n>.rds`, `latest_checkpoint()`, checkpoint callback runs last |
| 4 | `feat/resume-path` | the `path` learner parameter, `resume_training()`, `test_resume.R` |

### 1. `feat/ctx-callbacks`

New `ContextTorch$callbacks` field holding the active `CallbackSet`s by id, so a callback can reach
the others. `ctx$epoch` moves to before `on_begin`, so it is `0` rather than `NULL` there and a
callback can change where the loop starts. ~40 lines.

### 2. `feat/callback-state-dicts`

The three callbacks that carry state across epochs implement the `CallbackSet` extension point:

* lr_scheduler delegates to the wrapped `torch` scheduler. That scheduler only exists once the loop
  has begun, so a state loaded earlier is remembered and applied in `$on_begin()`.
* lr_one_cycle rejects a state saved for a schedule of a different length -- the policy is defined
  over total steps, so otherwise torch errors mid-run or the cycle never finishes annealing.
  (The plan originally put this check in branch 4; it belongs here, because `$load_state_dict()`
  is public API from this branch on.)
* early_stopping additionally stores `best_score` and `stagnation`.
* unfreeze records the trainable weights, restored both on load and at the end of `$on_begin()` --
  whichever runs last must win, or `$on_begin()` freezes them again.

`$state_dict()` is what lands in `learner$model$callbacks$<id>`, so the schedule and the trainable
weights are now part of the model.

### 3. `feat/checkpoint-state`

A checkpoint gains `state<n>.rds` next to `network<n>.pt` / `optimizer<n>.pt`, holding the epoch and
the other callbacks' state dicts. `saveRDS()` rather than `torch_save()`, which would strip classes
such as `data.table` off the history.

Two things found while implementing:

* **The checkpoint callback has to run last.** Callbacks execute in the order they were passed, so a
  checkpoint passed first saved the other callbacks *before* their own `on_epoch_end` -- a learning
  rate schedule one step behind the optimizer it was saved with. `train_loop()` now moves checkpoint
  callbacks to the end of the list.
* **No state file when the callbacks are past the checkpoint.** `$on_exit()` saves the last complete
  epoch after an interruption, but the other callbacks are already inside the epoch that was
  interrupted, so their states describe neither checkpoint. Resume falls back to the file suffix.

Plus `latest_checkpoint()` / `checkpoint_suffixes()` (skipping a checkpoint whose optimizer was
never written), and accepting a folder that already holds checkpoints as `path`.

### 4. `feat/resume-path`

The `path` train parameter -- a checkpoint folder, or `TRUE` to take the folder from this learner's
checkpoint callback -- plus `resume_training()`, called before the `on_begin` stage so callbacks see
the restored state. `epochs` is the total count, matching mlr3 hotstarting; an empty folder means
training starts from scratch, so one script serves both the first run and the restart. Warnings for
a checkpoint already at `epochs`, for callbacks that cannot restore, and for states of callbacks not
in this run. Also carries the `CallbackSetHistory` `rbind(fill = TRUE)` fix, only reachable here.
