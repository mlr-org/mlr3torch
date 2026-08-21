# TODO — `TaskTorch` branch

Open items for the general-purpose task type `"torch"` (`TaskTorch`, `PredictionTorch`,
`MeasureTorch`) and the supporting changes on this branch.
Ordered by how much they matter, not by effort.

Every item below was re-checked against the working tree. Anything that has since been fixed or
decided against was removed rather than carried along, so what is left here is open.

## 1. Design questions worth a second opinion

### 1.1 `predict_types` defaults to everything for every `"torch"` learner

The task is unknown at learner construction, so the learner cannot know whether probabilities are
available; `LearnerTorch$initialize()` falls back to `names(learner_predict_types$torch)`, i.e.
`c("response", "prob", "se")`. On a task whose encoder does not produce them,
`predict_type = "prob"` now yields a response-only prediction (`filter_predict_types()` drops what
was not asked for) and `mlr3` warns at score time, but a custom `prediction_encoder` that omits
`prob` still degrades silently.

*Suggestion, in increasing order of work:*

1. Document it and rely on the existing `mlr3` warning. Cheapest, and arguably in the spirit of the
   design.
2. Give `TaskTorch` a `predict_types` field (inferred from the target spec: `"prob"` only for
   `factor` and `logical` targets, or whatever the user declares alongside a custom
   `prediction_encoder`) and have `LearnerTorch` check the learner's `predict_type` against it at
   the start of `$predict()`. This turns a silent degradation into a clear error, and costs one
   field plus one check.
3. Default learners to `"response"` only and make `"prob"` opt-in. Safest, but it makes the common
   multi-label case need an extra argument, which cuts against the point of the quick route.

Option 2 looks like the right trade.

## 2. Known-good, do not re-investigate

Recorded so nobody spends time on these again.

**Decided, not defects.**

* **Address-based phashes** (`PipeOpTaskPreprocTorch`, `PipeOpModule` hashing
  `address(environment(self$fn))` / `address(self$module)`) break cross-session reproducibility.
  Won't fix: handing the function over to `hash_input` would trade the reproducibility break for
  `hash_input.function()`'s blindness to the closure environment.
* **Same-id `TorchDescriptor`s collide**, because `$phash` uses `class(self$generator)`, which for
  callbacks is always `R6ClassGenerator`. Accepted; the docstring already calls the phash "only
  heuristic".
* **A semantically mismatched learner and task runs silently.** The documented cost of one shared
  task type.
* **`lrn("torch.module")`'s dictionary prototype needs a dummy `loss`** (`R/LearnerTorchModule.R`),
  because there is no default loss for the `"torch"` task type and `as.data.table(mlr_learners)`
  builds prototypes. It is not a default at `lrn()` time — that still errors, by design.

**Environment.**

* **6 failing tests are pre-existing**, reproduced on a pristine checkout of the branch commit:
  4 × `test_LearnerTorchTabM.R` man-page lookups, and `test_nn_graph.R:209` /
  `test_PipeOpTorchBlock.R:37` deep-clone hash mismatches (`hash_input.nn_module()` uses
  `data.table::address()`, which a deep clone cannot preserve).
* **Anything spawning a worker -- `encapsulate("callr", ...)`, `future::multisession` -- fails
  under `pkgload::load_all()` when the installed `mlr3torch` is stale.** The worker loads the
  *installed* package, not the source tree, so it errors on whatever the branch changed since:
  `Must have formal arguments: predict_tensor` (`test_LearnerTorchModel.R:86`,
  `test_LearnerTorch.R:443` and `:454`) or
  `could not find function ".__LearnerTorchModule__.dataset"` -- `leanify_package()` moves R6
  methods into the namespace, so any newly added one looks missing. All three pass again once the
  package is reinstalled; reinstall before believing a worker-related failure.

**Verified correct, tested.**

* **Truth attachment.** Row-id alignment tested three ways — explicit `row_ids`, a permuted
  `row_roles$use`, and per-row versus batched prediction. `measures_train`, `measures_valid`,
  `t_clbk("history")`, `internal_valid_scores`, `predict_newdata`, `store_models`, `predict_sets`,
  marshaling and the `po("torch_model")` route all carry a correctly aligned truth. Alignment after
  fold combination is exact for `holdout`, `cv`, `repeated_cv`, `insample`, `bootstrap`,
  `subsampling` and `rsmp("custom")` with a reverse-sorted test set.
* **Zero-target tasks.** Train, predict, score, `resample()`, `$aggregate()`, `as.data.table()` and
  `$obs_loss()` all work; `truth` is *absent* rather than present-and-`NULL`, and stays absent
  through `c()`, `filter_prediction_data()` and `create_empty_prediction_data()`.
* **`TaskTorch$hash`.** Changes on `$filter()`, `$select()`, `$cbind()`, `$rbind()`,
  `$droplevels()`, col-role and id changes, all three configuration fields, and on mutating a
  deep-cloned `default_measure`. Stable across repeated calls, `$clone(deep = TRUE)`,
  reconstruction, `saveRDS`/`readRDS` and separate R sessions.
* **`Task` conformance.** All registered col roles work. `$truth()` respects row order, matching
  `TaskRegr` exactly. Task surgery, `$internal_valid_task`, `$print()`, `$formula()`,
  `as.data.table()`, `$missings()`, `$col_info` and all 28 `Task` active bindings compared side by
  side against `tsk("mtcars")` — no divergence. `resample()`/`benchmark()` with `stratum`, `group`,
  `order` and `name` roles all instantiate and aggregate correctly.
* **Tuning.** Non-internal tuning (`tune()`, `auto_tuner()`, nested resampling) and internal tuning
  (`tnr("internal")`, `to_tune(internal = TRUE)`, `validate = "test"`/`"predefined"`, `patience`,
  `msr("internal_valid_score")`), `store_models`, marshal round trip, `future::plan("multicore")`
  and an empty-test-set fold all behave.
* **Preprocessing.** Generic `mlr3pipelines` ops (`scale`, `pca`, `encode`, `removeconstants`,
  `imputemean`, `colapply`) and `mlr3torch`'s `lazy_tensor` ops (`trafo_*`, `augment_*`) all train
  on a `TaskTorch`, return a `TaskTorch` with targets, truth and `output_dim_for()` intact, and work
  inside pipelines and under `resample()`. Ops that interpret the target (`classbalancing`, `smote`,
  `ppl("targettrafo")`) correctly reject it.
