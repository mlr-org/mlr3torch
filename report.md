# `TaskTorch` audit — surprising behaviour

Scope: `TaskTorch` / `PredictionTorch` / `MeasureTorch` and the `LearnerTorch` code paths that
serve them (`learner_torch_predict`, `measure_prediction`, `create_empty_prediction_data`,
`get_batch_constructor`), plus `?TaskTorch`, `?PredictionTorch`, `?mlr_measures_torch` and
`vignettes/articles/custom_task_type.Rmd`.

Roughly 60 configurations were exercised across ~25 fresh `Rscript` sessions: six task shapes
(one numeric target, one `factor`, one `logical`, two `logical`, two numeric, a `lazy_tensor`
target, and no target at all), all four predict types including mismatches, single- and
multi-head networks, and the `$train()`/`$predict()`, `predict_newdata`, `resample()`,
`benchmark()`, `tune()`, `auto_tuner()`, internal tuning, validation/early-stopping and
`po("torch_model")` contexts. Everything below was reproduced at least twice, the second time
in a fresh process with independently written test code.

Overall impression: the *plumbing* is solid. Row-id alignment, fold combination, filtering,
measure weights, `saveRDS` round trips and the tuning/validation machinery all behaved. What
bites is the **measure layer** (`msr_torch()` accepts more than it can deliver, and silently
overrides argument defaults) and the **empty-prediction path**
(`create_empty_prediction_data.TaskTorch` synthesises a network output that does not match what
the encoder is ever handed in real life).

---

## 1. Silently wrong results

### 1.1 `msr_torch()` overrides a scoring function's default arguments with `NULL`

`mt_invoke()` (`R/MeasureTorch.R:76-83`) builds `args` containing *every* supported name and then
keeps whichever the function declares. `list(weights = prediction$weights)` keeps the name even
when the value is `NULL`, so `do.call()` passes `weights = NULL` **explicitly** and the default in
the user's function is never used.

```r
t = as_task_torch(d, target = "y", id = "reg", output_dim = function(task) 1L,
  default_encoder = enc1)                      # no `weights_measure` column role
l$train(t); p = l$predict(t)

m = msr_torch("wmse", function(truth, response, weights = rep(1, length(truth)))
  sum((truth - response)^2 * weights) / length(truth), range = c(0, Inf), minimize = TRUE)
p$score(m)
```

```
unweighted MSE by hand : 2.719837
msr_torch score        : 0
task has weights_measure: FALSE
```

Making weights optional with a default is the obvious way to write a measure that works on
weighted and unweighted tasks. Instead of the unweighted score you get `sum(... * NULL) / n` =
`0` — a plausible number, and a *perfect* one for a loss. The same holds for `prob`, `se`,
`task`, `learner` and `train_set`: a default value on any of them is dead.

### 1.2 A measure that declares `learner` gets `requires_learner`, never `requires_model`

`R/MeasureTorch.R:36-40` derives `requires_task` / `requires_learner` / `requires_train_set` from
the declared arguments. `mlr3` distinguishes `requires_learner` ("the learner, model or not")
from `requires_model` ("the trained model must be there"), and `msr_torch()` can never produce
the latter automatically. The only reason a *torch* measure wants the learner is the network, so
this is the common case.

```r
mnet = msr_torch("nparam", function(truth, response, learner) length(learner$network$parameters))
rr = resample(t, l, rsmp("holdout"), store_models = FALSE)
rr$aggregate(mnet)
```

```
auto properties: requires_learner
msr_torch version : 0
requires_model    : ERR:  ✖ Measure 'nparam2' requires the trained model → Class: Mlr3ErrorInput
```

`learner$network` is `NULL` when the model was not stored, so the measure quietly reports `0`
parameters. Passing `properties = "requires_model"` by hand gives `mlr3`'s clean error (second
line), which is what the automatic derivation should have produced. The doc template
(`man-roxygen/params_measure_torch.R`) lists the four properties that are added automatically and
says nothing about `requires_model`.

### 1.3 An `obs_loss` that returns one number is recycled over every observation

`MeasureTorch` validates the *names* of `obs_loss`'s arguments at construction
(`R/MeasureTorch.R:31-33`) but never the length of what it returns, and `mlr3` assigns the result
with `data.table::set()`, which recycles.

```r
ms = msr_torch("s", function(truth, response) 1,
  obs_loss = function(truth, response) mean((as.matrix(truth) - response)^2))
head(p$obs_loss(ms), 3)
```

```
   row_ids  truth.y1    truth.y2    response        s
     <int>     <num>       <num> <pt_arrays>    <num>
1:       1 0.3442853  0.01488106  <array[2]>  3.460117
2:       2 0.3237076 -2.59717421  <array[2]>  3.460117
3:       3 3.2984769 -1.10941796  <array[2]>  3.460117
```

Writing `mean(...)` where `rowMeans(...)` was meant — easy on a multi-target task, where the
per-observation loss is a row reduction rather than the identity — produces a per-observation
column that is a constant. An `obs_loss` returning a *matrix* does error (`Supplied 80 items to
be assigned to 40 items of column 'mse'`), so length 1 is the only silent case.

---

## 2. Wrong state surfacing as a confusing error

### 2.1 A task with **no target at all** cannot be trained without a dummy `target_batchgetter`

`R/LearnerTorchModel.R:152` is `private$.target_batchgetter %??% get_target_batchgetter(task)`.
`%??%` evaluates the right-hand side whenever the left one is `NULL`, and
`get_target_batchgetter.TaskTorch()` unconditionally `stopf()`s. So a target-less task plus a
learner with no `target_batchgetter` — exactly the unsupervised setup the vignette describes —
errors before `get_batch_constructor.default()` (`R/task_dataset.R:308-311`) gets a chance to
discard the batchgetter it would have discarded anyway.

```r
tu = as_task_torch(d[, .(x1, x2, x3)], id = "ae", output_dim = function(task) 3L,
  default_encoder = ...)
lu = lrn("torch.module", module_generator = ..., ingress_tokens = list(x = ingress_num()),
  loss = TorchLoss$new(nn_module("sq", forward = function(input) input$pow(2)$mean()), id = "sq"),
  epochs = 2, batch_size = 16)          # no target_batchgetter: there is no target
lu$train(tu)
```

```
n targets: 0
ERR: Task 'ae' does not define how its target becomes a tensor -- what `y` has to look like
follows from the loss, so it is the learner that decides. Pass `target_batchgetter` to the
learner (e.g. `lrn("torch.module")`) or overwrite the method for your own `LearnerTorch` subclass.
```

The message asks for something the task does not have. `custom_task_type.Rmd:307` says the
opposite: *"A task with no targets has no target element in its batches at all, so the loss is
called as `loss(y_hat)`, with no second argument at all."* The vignette's own autoencoder only
works because it happens to pass `target_batchgetter = function(data, x) x[[1L]]`.

The `po("torch_model")` route fails identically (`This happened in PipeOp torch_model's $train()`).
The workaround — `target_batchgetter = function(data) NULL` — trains fine, which confirms the
batchgetter is not actually needed.

### 2.2 An empty prediction hands a **single tensor** to an encoder that always sees a `list()`

`create_empty_prediction_data.TaskTorch()` (`R/PredictionTorch.R:221-222`) calls the encoder with
`torch_zeros(0L, output_dim_for(task))`. For a network with more than one head the encoder is
handed a *named list* of tensors on every real prediction — the docs promise exactly that
(`R/learner_torch_methods.R:504`, *"returns a `list()` of tensors, which is passed on
unchanged"*) — so the empty path is the one call where the contract is broken.

```r
mh = nn_module("mh", initialize = function(task) { self$h1 = nn_linear(3, 1); self$h2 = nn_linear(3, 1) },
  forward = function(x) list(m = self$h1(x), s = self$h2(x)))
enc_mh = function(task, network_output, predict_type) {
  stopifnot(is.list(network_output))
  list(response = as.numeric(network_output$m$cpu()))
}
lmh$train(tmh)
lmh$predict(tmh, row_ids = integer(0))
```

```
full predict ok: TRUE
empty predict  : ERR: is.list(network_output) is not TRUE
```

Whatever the encoder happens to throw is the whole message — no task, no learner, no hint that
this is the zero-row path. The same fires for an empty fold under `resample()`
(`Warning: Caught simpleError. Canceling all iterations ...`), i.e. far away from the mistake,
and only for some resamplings. An encoder that is *tolerant* of a bare tensor would instead
produce a silently misshaped empty prediction.

### 2.3 An empty prediction demands `output_dim` from a task that legitimately has none

Same line, other half. `output_dim` is documented as optional
(`custom_task_type.Rmd:93`: *"a network that sizes its own output ... never asks for it"*), and a
task without it trains and predicts perfectly — until something asks for a zero-row prediction.

```r
tno = as_task_torch(d, target = "y", id = "noodim", default_encoder = enc1)  # no output_dim
lno$train(tno)
lno$predict(tno, row_ids = integer(0))
```

```
full predict ok: TRUE
empty predict  : ERR: Task 'noodim' has no `output_dim`. Pass one to the task, or size the
network's output yourself with `nn("linear")` instead of `nn("head")`.
```

The advice is doubly confusing here: the user *did* size the network themselves, which is why
`output_dim` is absent. Reproduced through `resample()` with `rsmp("custom")` and an empty test
set as well.

### 2.4 An encoder that returns a tensor or `NULL` gets an opaque error

`check_prediction_data.PredictionDataTorch()` has careful messages for a bare `list()` and for a
row-count mismatch, but nothing checks that the encoder returned a named `list()` at all:

```
-- returns tensor not list : ERR: cannot coerce type 'externalptr' to vector of type 'list'
-- returns NULL            : ERR: attempt to set an attribute on NULL
```

`default_encoder = function(task, network_output, predict_type) network_output` is a plausible
first attempt, and neither message names the task, the learner or the encoder. (A misspelled or
unknown element name *is* caught well: `Names must be a subset of {'response','prob','se',
'lazy_tensor','extra','raw'}, but has additional elements {'reponse'}`.)

---

## 3. Inconsistencies

### 3.1 `msr_torch()` validates `obs_loss`'s arguments but not `fun`'s

```r
msr_torch("t", function(truth, respones) 1)                                # accepted
msr_torch("t", function(truth, response) 1, obs_loss = function(truth, respones) 1)
#> ERR: Assertion on 'arguments of `obs_loss`' failed: Must be a subset of
#>     {'truth','response','prob','se','prediction','task','learner','weights'} ...
```

`R/MeasureTorch.R:27` is a bare `assert_function(fun)`. The typo surfaces only at score time as
`argument "respones" is missing, with no default`, and inside a training run
`measure_prediction()` turns it into `Measure 'x' could not be computed and is reported as NaN`.

The same gap makes `predict_type = "lazy_tensor"` a trap: `assert_choice(predict_type,
pt_predict_types)` (`R/MeasureTorch.R:47`) accepts it, but `mt_invoke()` never builds a
`lazy_tensor` argument, so

```r
msr_torch("lt", function(truth, lazy_tensor) length(lazy_tensor), predict_type = "lazy_tensor")
#> at score time: ERR: argument "lazy_tensor" is missing, with no default
```

The doc template says `predict_type` is *"`"response"` (default), `"prob"` or `"se"`"*, so the
code is more permissive than the documentation in a direction that cannot work. (Reaching the
tensor through the `prediction` argument does work.)

### 3.2 `as.data.table()` gives a matrix `response` one `<array[k]>` column, not `response.*` columns

```r
head(as.data.table(p), 2)
```

```
   row_ids  truth.y1    truth.y2    response
     <int>     <num>       <num> <pt_arrays>
1:       1 0.3442853  0.01488106  <array[2]>
2:       2 0.3237076 -2.59717421  <array[2]>
```

and with `predict_type = "prob"` on a two-label task:

```
   row_ids truth.a truth.b    response    prob.a    prob.b
     <int>  <lgcl>  <lgcl> <pt_arrays>     <num>     <num>
1:       1   FALSE   FALSE  <array[2]> 0.5271820 0.6475327
```

So three columns of the same table follow three different rules: a `data.table` `truth` spreads
into `truth.*`, a `prob` matrix spreads into `prob.*`, and a `response` matrix collapses into
one cell per observation. This is deliberate and tested
(`tests/testthat/test_PredictionTorch.R:286` asserts `expect_false(any(startsWith(names(tab),
"response.")))`), but `custom_task_type.Rmd:185` states the opposite: *"The `truth.*` columns are
the ground truth taken from the task and the `response.*` columns are what the network
predicted."* Worth noting that `R/PredictionTorch.R:288-291` makes the `is.matrix(el)` branch
unreachable for anything but `prob`, since `is.array()` is true for a matrix and `pt_arrays()`
runs first — so the branch reads as if matrices were meant to spread.

### 3.3 `prediction$score()` with no arguments cannot use the task's `default_measure`

```r
tdm = as_task_torch(d, target = "y", default_measure = msr_torch("mse", ...))
pdm = l$predict(tdm)
pdm$score()
```

```
ERR:
✖ Measure 'torch.default' requires a task
→ Class: Mlr3ErrorInput
```

`?TaskTorch` describes `default_measure` as *"The default measure of the task, which is e.g.
used in `prediction$score()`"* (`man/mlr_tasks_torch.Rd:126`). It is not: a bare `Prediction` has
no task, so `msr("torch.default")` cannot resolve. `pdm$score(msr("torch.default"), task = tdm)`
and `rr$aggregate()` both work; only the documented no-argument form does not.

### 3.4 Missing predictions are reported for a vector response and never for anything wider

```
matrix response, 3 all-NA rows -> $missing: 0
vector response, 3 NA rows      -> $missing: 1 2 3
```

`is_missing_prediction_data.PredictionDataTorch()` (`R/PredictionTorch.R:158-166`) bails out for
anything that is not a dimensionless atomic vector. The reasoning is in a source comment, but
neither `?PredictionTorch` nor the vignette mentions it, so the same NaN blow-up is visible on a
single-target task and invisible on a multi-target one.

### 3.5 `msr("torch.default")$obs_loss`'s error message is unreachable

`MeasureTorchDefault` writes a careful `stopf()` in `.obs_loss` (*"does not delegate the
per-observation loss ..."*), but it never sets the `"obs_loss"` property, so `mlr3` never calls
it and `prediction$obs_loss()` returns a column of `NA` instead:

```
   row_ids truth.a truth.b    response torch.default
1:       1   FALSE   FALSE  <array[2]>            NA
```

That silent `NA` column matches what `mlr3` does for any measure without the property
(`msr("classif.auc")` behaves the same), so this is only a dead-message inconsistency — but the
author clearly intended the error to fire.

---

## Checked and found unsurprising

Do not redo these.

**Task shapes.** One numeric target, one `factor`, one `logical`, two `logical`, two numeric, a
`lazy_tensor` target, and no target at all. For each: `$train()`, `$predict()`, `$score()`,
`as.data.table()`, and a 2- or 3-fold `resample()`. `truth`, `response` and `prob` come back with
the right class every time; a `lazy_tensor` target survives fold combination with its rows in the
right places (checked numerically against `task$truth()` via `match()` on row ids).

**Predict types.** Opt-in is enforced (`predict_types = "response"` by default; setting
`predict_type = "prob"` errors with *"Learner 'torch.module' does not support predict type
'prob'"*). Switching `predict_type` after training works. An encoder that does not produce `prob`
under `predict_type = "prob"` degrades to a response-only prediction plus `mlr3`'s *"Measure 'x'
is missing predict type 'prob' of prediction"* warning (this is TODO §1.1). An encoder that
always produces `prob` puts it in the prediction even when the learner promised only `response`.
`se` works end to end (two-head network, `se` measure, `as.data.table()` shows a `se` column).
`lazy_tensor` from a multi-head network is refused with the intended message; `lazy_tensor`
predictions combine correctly across CV folds and `materialize()` reproduces the encoder's input
exactly.

**Contexts.** `predict_newdata` with and without the target columns present (targets become
`NA`); `resample()` with `cv`, `holdout` and `rsmp("custom")` including an empty test fold (works
whenever `output_dim` is present and the network has one head); `benchmark()` over two tasks ×
two learners; `tune()` with `tnr("random_search")`; `auto_tuner()` inside `resample()`; internal
tuning with `epochs = to_tune(upper = 10, internal = TRUE)`, `tnr("internal")`, `patience` and
`measures_valid`; `validate = 0.3` with early stopping (`internal_valid_scores` and
`internal_tuned_values` both populated); `po("scale") %>>% ... %>>% po("torch_model")` as a
`GraphLearner`, trained, scored, and switched to `predict_type = "prob"`.
`tune(measure = msr("torch.default"))` errors cleanly on `minimize = NA`.
`measures_train`/`measures_valid` reject learner-requiring measures at construction with
*"Measures must not require a learner or model"*.

**Predictions.** Fold combination is aligned (`lazy_tensor` and matrix responses both checked by
row id). `$filter()`, `c()` (including the *"Cannot combine prediction data with different
predict types: truth, response vs truth, lazy_tensor"* guard), `saveRDS`/`readRDS` of a
`PredictionTorch` and of a whole `ResampleResult` (scoring afterwards works), empty predictions
with a single-head network, and `check_prediction_data`'s messages for a bare `list()` response
and for a row-count mismatch — all fine.

**Measure weights.** `weights_measure` reaches the measure correctly and stays aligned:
`sum(weights)` was 820 on the full task, 6 after `$filter(1:3)`, and 820 again after combining
two CV folds.

**Measure arguments.** A `fun` declaring all nine supported arguments received
`truth = data.table`, `response = matrix`, `prob = matrix`, `se = NULL`,
`prediction = PredictionTorch`, the right `task`, a `LearnerTorchModule`, a 40-element
`train_set` and `weights`. `msr("torch.default")` delegates `task`, `learner` and `train_set`
through to the task's measure, so a `default_measure` declaring `train_set` scores correctly via
`rr$aggregate()`. A measure returning a non-scalar errors.

**Not re-investigated** (already in `TODO.md`): `lazy_tensor` predictions and `saveRDS`, the
missing fallback learner for `"torch"`, semantically mismatched learner/task pairs, address-based
phashes, and worker-based encapsulation under `pkgload::load_all()`.

One thing deliberately left as a documented design cost rather than a finding:
`bmr$aggregate()` over two tasks with *different* `default_measure`s puts both numbers in one
column called `torch.default` (`0.4230769` Hamming next to `1.6358767` MSE). The vignette's
*"What You Give Up"* section already says `mlr3` cannot tell one custom problem from another.
