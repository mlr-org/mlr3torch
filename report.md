# TaskTorch review findings

Adversarial review of the `TaskTorch` machinery at `3d9e45104`, branch `vignette/custom-task-type`.

Four agents were run, one per mechanism: truth attachment, hashing, `TaskTorch` as a plain
`Task`, and resample/benchmark/tuning. Every finding below was reproduced by an agent and then
re-verified independently before being written down. **Nothing has been fixed yet.**

Provenance column: **new** = introduced by this branch, **exposed** = pre-existing code that only
becomes reachable through `TaskTorch`, **upstream** = mlr3/mlr3misc behaviour.

---

## Severity 1 — silently produces a wrong number

These do not error. They return something plausible.

### 1.1 `msr("torch.default")` hard-codes the optimisation direction — tuning selects the worst configuration · new · **FIXED**

`mlr_reflections$default_measures$torch` is `"torch.default"`, so `MeasureTorchDefault` is what
`tune()`, `AutoTuner`, `$aggregate()` and early stopping fall back to. It delegates `.score()` to
`task$default_measure` but not the metadata that decides direction: `minimize = TRUE` and
`range = c(-Inf, Inf)` are constants in `R/MeasureTorch.R`.

```r
acc = msr_torch("acc", function(truth, response) mean(as.matrix(truth) == as.matrix(response)),
  minimize = FALSE, range = c(0, 1))
task = tt_task(d, target = c("a", "b"), default_measure = acc)
acc$minimize             #> FALSE
msr("torch.default")$minimize  #> TRUE   <- what the tuner believes
```

End to end, a grid search over `epochs` reports `epochs=1 -> 0.475`, `epochs=20 -> 0.500` and
returns `epochs = 1`. The tuner maximises nothing and minimises an accuracy.

`regr`/`classif` cannot reach this: their `default_measures` point at a concrete measure whose
`minimize` matches what it computes.

**Fixed.** `minimize` and `range` are now construction arguments of `MeasureTorchDefault`
defaulting to `NA` / the unbounded range, i.e. "not known until a task is in sight". mlr3 refuses to
tune with an `NA` direction ("Measure torch.default has its `minimize` field set to NA, which is
disallowed when tuning"), so the silent inversion is now a loud refusal. Passing the direction
explicitly (`msr("torch.default", minimize = FALSE, range = c(0, 1))`) tunes correctly, and
`.score()` errors if the stated direction contradicts the task's measure. Scoring and
`$aggregate()` are unaffected, since neither consults `minimize`.

### 1.2 Factory-built closures collide, and `benchmark()` then scores a row with another row's task · exposed

`hash_input.function()` is `list(formals(x), as.character(body(x)))` — the closure environment is
deliberately ignored. `TaskTorch` and `MeasureTorch` are the first surfaces that hash
*user-supplied* closures, so this now has teeth.

```r
make_odim = function(k) function(task) k
t1 = tt_task(d, target = "y", id = "t", output_dim = make_odim(1L))
t2 = tt_task(d, target = "y", id = "t", output_dim = make_odim(3L))
t1$hash == t2$hash        #> TRUE   (output_dim_for gives 1 vs 3)

bmr = benchmark(benchmark_grid(list(t1, t2), learner, rsmp("holdout")))
nrow(bmr$tasks)           #> 1
bmr$score(m)$odim         #> 1, 1   <- second row should be 3
```

`resample(t2, ...)$aggregate(m)` alone gives the right answer; only `benchmark()` collapses them.
Confirmed identically for `prediction_encoder`, `MeasureTorch`'s `fun` and its `obs_loss`,
`LearnerTorchModule`'s `target_batchgetter`, and factory-built `nn_module` generators.

**Fix.** Hash the closure itself (`digest` includes the environment and was verified session-stable
even with a torch tensor or an R6 task captured), or require `crate()` and fold
`as.list(environment(x))` into a local `hash_input` method.

### 1.3 `as.character(body(x))` drops argument names · upstream

A body that is a single call loses its argument *names*, which is the common shape for a
`prediction_encoder` or `output_dim`.

```r
as.character(quote(f(a = 1, b = 2)))   #> "f" "1" "2"
as.character(quote(f(b = 1, a = 2)))   #> "f" "1" "2"   <- identical
```

So an encoder returning `list(response = x, prob = NULL)` and one returning
`list(prob = x, response = NULL)` share a hash while producing different predictions. A `{ }` body
is safe; one-liners are not.

**Fix.** Upstream: `deparse(body(x))` keeps the names. Locally, 1.2's fix also covers this.

### 1.4 `target_batchgetter` is absent from `PipeOpTorchModel`'s phash · exposed

`PipeOpTorchModel$.additional_phash_input()` returns only `private$.task_type`, discarding
`PipeOpLearner`'s `private$.learner$phash`; `LearnerTorchModel` omits the batchgetter too.

```r
po("torch_model", target_batchgetter = f1)$phash ==
  po("torch_model", target_batchgetter = f2)$phash     #> TRUE
```

The batchgetter decides what `y` *is*, so two GraphLearners optimising different targets are one
object to mlr3: `c(as_benchmark_result(rr1), as_benchmark_result(rr2))` collapses to a single
learner row, and anything reading `bmr$learners$learner` or a `requires_learner` measure gets the
wrong one. `LearnerTorchModule` *does* include it — the sibling classes disagree.

**Fix.** `list(private$.task_type, private$.learner$phash)` in `PipeOpTorchModel`, and add
`private$.target_batchgetter` to `LearnerTorchModel`.

### 1.5 `weights_measure` is registered but unsupported, and misaligns silently · new

`task_properties$torch` includes `weights_measure` and `measure_properties$torch` includes
`weights`, but `PredictionTorch$new()` has no `weights` argument and `pt_elements` omits it.

```r
task$set_col_roles("w", "weights_measure")
learner$predict(task)
#> Error in initialize(...): unused argument (weights = c(0.82, 0.24, ...))
```

It also breaks training as soon as `measures_valid` runs. Worse, when a `weights` element does
reach the S3 methods it is silently corrupted:

```r
filter_prediction_data(pdata, row_ids = c(2L, 4L))
#> row_ids 2 4 | response 2 4 | weights 10 20 30 40 50   <- not subset, now misaligned
names(c(pdata, pdata))
#> row_ids, truth, response                              <- weights dropped
```

`check_prediction_data()` catches neither. `create_empty_prediction_data.TaskTorch()` also never
sets `weights = numeric()`, unlike the regr/classif methods.

**Fix.** Either make `weights` first class (constructor argument; include it in the set that
`check_prediction_data`, `c.PredictionDataTorch` and `filter_prediction_data` iterate over — it is
plain numeric, so `pt_subset`/`pt_combine` already handle it), or drop `weights_measure` and
`weights` from the reflections so the configuration is rejected up front. The same asymmetry
applies to `extra`/`raw`, which mlr3 permits and `PredictionTorch$new()` cannot take.

### 1.6 Validation measures receive the validation rows as `train_set` · exposed

`measure_prediction()` uses `train_set = task$row_roles$use`, and for the validation call that task
is `ctx$task_valid`.

```r
m = msr_torch("nts", function(truth, response, train_set) length(train_set))
# 40 rows, validate = 0.25
#>    epoch train.nts valid.nts
#> 1:     1        30        10        <- valid.nts should also be 30
```

Not `TaskTorch`-specific — the same wiring runs for classif/regr, but stock mlr3 measures rarely
request `train_set`, so it was unobservable until `msr_torch()` made it easy to ask for.

**Fix.** Give `measure_prediction()` an explicit `train_set` argument and pass
`ctx$task_train$row_roles$use` from both call sites in `R/learner_torch_methods.R`.

---

## Severity 2 — wrong state that surfaces as an opaque error

### 2.1 Duplicated target names are accepted · new

`TaskTorch$initialize()` calls `assert_character(target, any.missing = FALSE, null.ok = TRUE)`
without `unique = TRUE`. `TaskRegr`/`TaskClassif` cannot reach this state because their
`task_check_col_roles` methods cap `target` at one column, and the `task_check_col_roles.TaskTorch`
method that used to sit in the chain was deleted with the `TaskSupervised` change.

```r
t = tt_task(d, target = c("y", "y"))
t$target_names     #> "y" "y"
t$ncol             #> 5     (the task has 4 columns)
output_dim_for(t)  #> 2     (there is 1 target column)
t$truth()          #> Error ... 'cols' failed: Must have unique names
as_task_regr(d, target = c("y", "y"))   #> rejected
```

A network is sized wrong before anything errors, and the eventual error names neither the task nor
the target.

**Fix.** `unique = TRUE` in the assertion, plus the same check in a `task_check_col_roles.TaskTorch`
so the role cannot be duplicated after construction.

### 2.2 `prediction_encoder` and `default_measure` are unvalidated public fields · new

`output_dim` is an active binding that asserts; its two siblings are plain fields assigned only at
construction.

```r
t$output_dim = 42          #> Error: Must be a function (or 'NULL')   <- good
t$prediction_encoder = 42  #> accepted
t$default_measure = 42     #> accepted
t$hash                     #> Error: $ operator is invalid for atomic vectors
```

`$hash` is touched by `resample()`, `benchmark()` and caching.

**Fix.** Make both active bindings that re-run the constructor's assertions.

### 2.3 `PipeOpTorchFn` drops everything `PipeOpTorch` hashes · exposed

`.additional_phash_input()` returns only `private$.fn`, overriding the `PipeOpTorch` version that
includes `private$.shapes_out_fn`, channel names and `param_set$ids()`.

```r
q1 = po("nn_fn", fn = function(x) x, shapes_out = function(...) list(c(NA, 1L)))
q2 = po("nn_fn", fn = function(x) x, shapes_out = function(...) list(c(NA, 7L)))
q1$phash == q2$phash   #> TRUE, while the inferred shapes are NA x 1 vs NA x 7
```

**Fix.** `c(super$.additional_phash_input(), list(private$.fn))`.

### 2.4 `weights_learner` is registered but no `LearnerTorch` can declare `"weights"` · new

`task_col_roles$torch` and `task_properties$torch` include `weights_learner`, but
`learner_properties$torch` is `c("validation", "internal_tuning", "marshal")`, and `Learner$new()`
asserts properties against that registry — so the property is unreachable by construction. Nothing
in `R/` reads `task$weights_learner`.

Every `TaskTorch` with such a column fails to train (loudly, hence severity 2); the only way
forward is `use_weights = "ignore"`, which trains while silently discarding the weights.

**Fix.** Drop the role from the registry, or add `"weights"` to `learner_properties$torch` and
consume `task$weights_learner` in the training loop.

---

## Severity 3 — real but narrow

### 3.1 `msr("torch.default")` never reports a per-observation loss · new

`MeasureTorchDefault` delegates `.score()` but declares no `"obs_loss"` property and no
`.obs_loss()`, so the delegation stops at scoring.

```r
rr$obs_loss(m)$mse            #> 1.755, 0.560
rr$obs_loss()$torch.default   #> NA, NA
```

Contradicts the `MeasureTorch` docs, which say `obs_loss` is what `$obs_loss()` and a
`ResampleResult`'s table report.

**Fix.** Include `"obs_loss"` in the properties and forward `.obs_loss()` to
`task$default_measure`. The property is fixed at construction while the default measure is
per-task, so this needs an active binding on `$properties` or an unconditional property with an
`NA` fallback.

### 3.2 Predict-type filtering is asymmetric between the empty and the real prediction · new

`create_empty_prediction_data.TaskTorch()` filters the encoder output to the learner's
`predict_type`; `learner_torch_predict()` does not filter at all.

```r
learner$predict_type      #> "response"
pred$predict_types        #> "response" "prob"
c(create_empty_prediction_data(task, learner), pred$data)
#> Error: Cannot combine prediction data with different predict types
```

Needs an encoder that ignores its `predict_type` argument, which the docs advise against, and only
bites on the failed-fold path.

**Fix.** Filter in both places or neither.

### 3.3 Address-based phash breaks cross-session reproducibility · exposed

`PipeOpTaskPreprocTorch` and `PipeOpModule` hash `address(environment(self$fn))` /
`address(self$module)`, so a `GraphLearner` containing any lazy-tensor preprocessing gets a
different hash in every session, and two structurally identical `PipeOpModule`s never agree —
`p1$phash != p1$clone(deep = TRUE)$phash`.

The justifying comment in `R/PipeOpModule.R` — *"mlr3pipelines does not use calculate_hash, but
calls directly into digest"* — is false: `PipeOp$phash` is
`calculate_hash(class(self), self$id, private$.additional_phash_input())`.

**Fix.** Hand the function/module over and let `hash_input.function` /
`hash_input.nn_module_generator` handle it. That trades the reproducibility break for the lesser
collision in 1.2.

### 3.4 Same-id descriptors collide · exposed

`TorchDescriptor$phash` uses `class(self$generator)`, which for callbacks is always
`R6ClassGenerator`.

```r
torch_callback("cb", on_epoch_end = function() 1)$phash ==
  torch_callback("cb", on_epoch_end = function() 2)$phash   #> TRUE
```

Two custom `nn_module` losses with the same id and no parameters collide too. The docstring calls
the phash "only heuristic", but the heuristic fails for exactly the case the docs invite.

**Fix.** Hash `self$generator` rather than `class(self$generator)`.

### 3.5 `as_task_torch()` cannot coerce an existing `Task` · new

```r
as_task_regr(tsk("mtcars"))                            #> TaskRegr
as_task_torch(tt_task_labels(10), target = c("a","b"))
#> Error: no applicable method for 'as_data_backend' applied to class "TaskTorch"
```

Round-tripping a `TaskTorch` to attach a different `output_dim` is impossible. Errors rather than
misbehaves.

**Fix.** Accept a `Task` by taking its `backend` and roles, or document the limitation.

---

## Correction to `3d9e45104`

The commit message said the removed `hash_input()` calls "just apply it twice". That is wrong for
one site: dropping `lapply(private$.ingress_tokens_, hash_input)` in `LearnerTorchModule` stopped
`hash_input.function` from being applied to a `Selector`, which had been collapsing
`ingress_ltnsr(NULL)` and `ingress_ltnsr("a")` to one hash. They now differ. The code is right; the
message understated it.

---

## Verified clean

Worth recording, because several of these were the suspected weak points.

**Truth attachment.** Row-id alignment tested three ways — explicit `row_ids`, a permuted
`row_roles$use`, and per-row versus batched prediction. `Task$data()` preserves the requested row
order and `.index` is a position into `task$row_ids`, so alignment survives `shuffle = TRUE` and
custom samplers; `.dataloader_predict()` forces `shuffle = FALSE`. `measures_train`,
`measures_valid`, `t_clbk("history")`, `internal_valid_scores`, `predict_newdata`,
`store_models`, `predict_sets`, marshaling and the `po("torch_model")` route all carry a correctly
aligned truth. No callback builds predictions.

**Zero-target tasks.** Train, predict, score, `resample()`, `$aggregate()`,
`as.data.table()` and `$obs_loss()` all work; `truth` is *absent* rather than present-and-`NULL`,
and stays absent through `c()`, `filter_prediction_data()` and `create_empty_prediction_data()`.

**`TaskTorch$hash`.** Changes on `$filter()`, `$select()`, `$cbind()`, `$rbind()`,
`$droplevels()`, col-role and id changes, all three configuration fields, and on mutating a
deep-cloned `default_measure`. Stable across repeated calls, `$clone(deep = TRUE)`, reconstruction,
`saveRDS`/`readRDS` and separate R sessions.

**`Task` conformance.** All eight registered col roles work; `offset` — removed from the registry —
is rejected cleanly both ways. `$truth()` respects row order, matching `TaskRegr` exactly. Task
surgery, `$internal_valid_task`, `$print()`, `$formula()`, `as.data.table()`, `$missings()`,
`$col_info`, and all 28 `Task` active bindings compared side by side against `tsk("mtcars")` — no
divergence. `resample()`/`benchmark()` with `stratum`, `group`, `order` and `name` roles all
instantiate and aggregate correctly.

**Pipelines.** `po("scale")`, `po("pca")`, `po("removeconstants")`, `po("encode")`,
`po("colapply")` return a `TaskTorch` with targets, truth and `output_dim_for()` intact. The
documented claim that `PipeOpTaskPreproc` sees a `TaskTorch` as targetless is accurate.

---

## Addendum — resample/benchmark/tuning agent

That agent independently reproduced 1.1, 1.2, 1.5 and 3.1, and added the following. Its 1.1 repro
is stronger than the one above: with a measure made deterministic in `epochs`, tuning with the
measure itself returns `epochs = 3` and tuning with `msr("torch.default")` returns `epochs = 1`,
from the same archive.

It also notes that `tests/testthat/test_TaskTorch.R:411` ("the hash of a measure covers its scoring
function") asserts exactly the 1.2 benchmark scenario and passes, because its two measures have
literally different bodies. That test currently documents the gap as covered.

### A.1 `pt_combine()` silently turns factor levels the first fold did not see into `NA` · new · SEVERITY 1

`pt_combine()` does `factor(unlist(lapply(xs, as.character)), levels = levels(x))`, taking the
levels of the *first* non-empty element. mlr3's own `c.PredictionDataClassif` goes through
`rbindlist`, which unions levels.

```r
a = pdata(1:3, response = factor(c("a","b","a"), levels = c("a","b")))
b = pdata(4:6, response = factor(c("c","b","c"), levels = c("b","c")))
c(a, b)$response
#> a b a <NA> b <NA>        Levels: a b        <- 2 of 6 destroyed
# rbindlist on the same input unions to: a b c
```

End to end through `resample()` with folds that see disjoint classes: per-fold accuracy 1 and 1,
combined accuracy `NA`, 8 of 32 responses `NA`. With an `na.rm = TRUE` measure the combined score
comes out as a perfectly plausible **1** while a quarter of the predictions have been destroyed.

This is the same class of bug as the `unlist()` one that `pt_combine()`'s fallback was written to
fix, in the branch directly above it. A CV fold that happens not to observe a rare class is enough
to trigger it — no exotic setup required.

**Fix.** `levels = unique(unlist(lapply(xs, levels)))`, or `stopf()` when the level sets are not
equal, the way `pt_bind_arrays()` already errors on incompatible dimensions.

### A.2 Every error inside a validation measure becomes a silent `NaN` · exposed · SEVERITY 1

`measure_prediction()` wraps scoring in `tryCatch(..., error = function(e) NaN)`, discarding the
condition.

```r
learner = tt_learner(..., measures_valid = msr("torch.default"), validate = 0.3)  # task has no default_measure
learner$train(task)
learner$internal_valid_scores   #> $torch.default [1] NaN
```

The real error ("Task 'B' has no default measure") is thrown away with no warning. With
`patience > 0` the run then trains every epoch emitting only "Difference between subsequent
validation performances is NA"; with `epochs = to_tune(internal = TRUE)` the internal tuner
optimises a constant `NaN`. Confirmed for classif too, so not `TaskTorch`-specific.

**Fix.** Propagate the condition, or `warningf()` with the original message before degrading.

### A.3 A task with zero targets still demands a `target_batchgetter` · new · SEVERITY 2

`get_target_batchgetter.TaskTorch()` errors unconditionally, without checking
`length(task$target_names)`.

```r
t0 = tt_task(tt_data(20), target = character(0), id = "unsup", output_dim = function(task) 3L)
get_target_batchgetter(t0)
#> Error: Task 'unsup' does not define how its target becomes a tensor ...
```

The *Tasks without a Target* section of `?TaskTorch` says such batches simply have no `y`, so a
user following the documentation hits a misleading error and has to pass a dummy batchgetter.
(`get_batch_constructor.default()` does handle the targetless case — it is only this generic that
refuses first.)

**Fix.** Return `NULL` when `!length(task$target_names)`.

### A.4 Encapsulation is unusable on a `TaskTorch` — no fallback learner exists · new · SEVERITY 3

```r
resample(task, learner, rsmp("cv", folds = 2), encapsulate = "evaluate")
#> Error: Could not find default fallback learner for learner 'torch.module'
```

mlr3 requires a fallback for any encapsulation mode and nothing of task type `"torch"` is
registered to serve as one, so a `benchmark()` over `TaskTorch` tasks cannot be made robust to a
single failing row. Classif/regr torch learners have the featureless fallback. Errors loudly.

**Fix.** Register a `"torch"`-typed featureless learner, or add a `default_fallback()` method.

### A.5 `MeasureTorchDefault` also hardcodes `predict_type` · new · SEVERITY 3

Extends 3.1. `MeasureTorchDefault$predict_type` is fixed at `"response"`, so a `default_measure`
requiring `"prob"` reached through `torch.default` bypasses mlr3's up-front compatibility check and
yields `NaN` plus a "missing predict type 'prob'" warning instead of a clean refusal.

### Additionally verified clean by this agent

truth/response alignment after fold combination is *exact* for `holdout`, `cv`, `repeated_cv`,
`insample`, `bootstrap`, `subsampling` and `rsmp("custom")` with a deliberately reverse-sorted test
set — zero misaligned rows in all seven, tested with an identity network so `response == truth`.
Non-internal tuning (`tune()`, `auto_tuner()`, nested resampling), internal tuning
(`tnr("internal")`, `to_tune(internal = TRUE)`, `validate = "test"`/`"predefined"`, `patience`,
`msr("internal_valid_score")`), `store_models`, marshal round trip, `future::plan("multicore")`,
the `po("torch_model")` GraphLearner inside `resample()`/`benchmark()` and behind `po("scale")`,
`stratum`/`group` roles, and an empty-test-set fold all behave. `benchmark()` over tasks whose
measures *do* hash differently scores each row with its own measure.
