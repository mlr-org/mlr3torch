# `restore_best_weights` silently does nothing after a resume

Status: **resolved via option 2 (warn)**. Pre-existing on `feat/checkpoint-resume`; not introduced
by the merge of `main`, and not introduced by the callback `weight` change. Found while adding the
per-callback resume tests.

`CallbackSetEarlyStopping$load_state_dict()` now warns when `restore_best_weights` is set, so the
run no longer ends with a network that does not match the epoch it reports without saying so. The
weights are still not stored -- option 1 would put a full copy of the network into every
`state<n>.rds`, which with `freq = 1` is one copy per epoch on disk. The behaviour is documented in
the *Early Stopping* section of `man-roxygen/paramset_torchlearner.R` and pinned by
`tests/testthat/test_resume.R`. The rest of this file is kept as the record of why.

## Summary

`lrn(..., restore_best_weights = TRUE)` restores the best epoch's network at the end of a normal
run, but not at the end of a run that was resumed from a checkpoint. The resumed run reports a best
epoch from the previous run while holding the weights of a different epoch, with no error and no
warning.

## How the feature works normally

`CallbackSetEarlyStopping` clones the network whenever the validation score improves
(`private$.remember_weights()` → `self$best_state_dict`), and `$on_exit()` loads that copy back into
the network at the very end of training. A finished run therefore ends holding the best epoch's
weights, and reports that epoch through `$internal_tuned_values`.

## What breaks on resume

`CallbackSetEarlyStopping$state_dict()` — what gets written into `state<n>.rds` and read back by a
resuming run — stores only

```r
list(
  best_epochs = self$epoch_at_best_score,
  best_score  = self$best_score,
  stagnation  = self$stagnation
)
```

`best_state_dict` is not part of it, and `$load_state_dict()` does not set it. A resumed run
therefore starts with `best_state_dict = NULL`. If it never beats the restored `best_score`,
`.remember_weights()` is never called, `best_state_dict` stays `NULL`, and `$on_exit()` returns
early without restoring anything.

## Observable failure

Verified on `tsk("iris")` with `internal_valid_task = 1:30`, `patience = 10`,
`restore_best_weights = TRUE`, `opt.lr = 0.5`, `seed = 2`:

| run | reports `best_epochs` | weights actually in the model |
| --- | --- | --- |
| uninterrupted, 4 epochs | 1 | epoch 1 (restore works) |
| 2 epochs, then resumed to 4 | 1 | neither epoch 1 nor epoch 2 |

Under an `AutoTuner` doing internal tuning this archives the wrong model: `$internal_tuned_values`
names one epoch and the network is from another.

## Why the weights are not simply already on disk

The checkpoint is written in `$on_epoch_end()` / `$on_end()`, before the restore in `$on_exit()`.
That ordering is deliberate — it is what the `Inf` weight on `CallbackSetEarlyStopping` is for — so
that a checkpoint holds the network as *training* left it and can be continued from, rather than the
restored one. Consequently `network<n>.pt` is always the network at the end of epoch `n`, never the
best one.

With `freq = 1` the best epoch's weights do happen to be in the folder as
`network<best_epochs>.pt`, but this is not reliable: `freq > 1` can skip the best epoch, and
`CallbackSetEarlyStopping` has no knowledge of the checkpoint folder's path.

## Options

1. **Store the weights.** Add `best_state_dict` to `CallbackSetEarlyStopping$state_dict()` when
   `restore_best_weights` is `TRUE`. The feature then survives a resume. Cost: each `state<n>.rds`
   grows by a full copy of the network — only for users who opted into `restore_best_weights`, who
   already keep that copy in memory.
2. **Warn.** On resume, if `restore_best_weights` is active but the restored state carries no
   weights, warn that the previous run's best weights are unavailable. No size cost, but the feature
   still does not survive a resume.
3. **Document only.** Note the limitation in the *Resuming* section of
   `man-roxygen/paramset_torchlearner.R` and leave the behaviour as is.
4. **Read the best epoch back from the checkpoint folder.** Load `network<best_epochs>.pt` when it
   is present. Avoids duplicating the network, but couples `CallbackSetEarlyStopping` to
   `CallbackSetCheckpoint` and still needs a fallback for when `freq` skipped that epoch.

## Reproduction

```r
library(mlr3torch)
task = tsk("iris")
task$internal_valid_task = 1:30
path = tempfile()

make = function(epochs, cbs = list()) {
  lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, seed = 2,
    validate = "predefined", measures_valid = msr("classif.ce"), patience = 10L,
    min_delta = 0, restore_best_weights = TRUE, opt.lr = 0.5, callbacks = cbs)
}
weights = function(learner) as.numeric(learner$network$parameters[[1L]]$flatten())[1:3]

# a run that is checkpointed and then continued
make(2L, t_clbk("checkpoint", freq = 1, path = path))$train(task)
resumed = make(4L)
resumed$param_set$set_values(path = path)
resumed$train(task)

resumed$model$callbacks$early_stopping$best_epochs             # 1
names(readRDS(file.path(path, "state2.rds"))$callbacks$early_stopping)
# "best_epochs" "best_score" "stagnation" -- no weights

best = torch::torch_load(file.path(path, "network1.pt"))[[1L]]$flatten()
isTRUE(all.equal(weights(resumed), as.numeric(best)[1:3]))     # FALSE
```
