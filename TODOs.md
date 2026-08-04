# TODOs — `TaskTorch` branch

Open items for the general-purpose task type `"torch"` (`TaskTorch`, `PredictionTorch`,
`MeasureTorch`) and the supporting changes on this branch.
Ordered by how much they matter, not by effort.
Each item carries a suggested approach; none of them are implemented.

All correctness defects found so far — including the four raised by an adversarial review, the
silent factor mislabelling and `c.PredictionDataTorch()` dropping `prob` — are fixed and covered by
regression tests. `R CMD check` is clean and the test suite shows only pre-existing failures
(see section 4).

## 1. Usability warts

None of these corrupt results; they all produce a confusing failure later instead of a clear one at
the point of the mistake. All four are small and independent, so they make a decent single cleanup
commit.

### 1.1 Public fields bypass their constructor validation

`target_batchgetter`, `prediction_encoder` and `measure` are plain public fields, so
`task$measure = "not a measure"` is accepted and blows up much later.

*Suggestion:* convert all three to active bindings backed by `private$.target_batchgetter` etc.,
running the same asserts as `$initialize()`. The constructor then assigns through the binding rather
than duplicating the checks. Two things to watch:

* `$hash` reads these fields live, which is what makes post-construction mutation hash correctly —
  keep that property; a binding reading the private field preserves it.
* `private$deep_clone()` currently special-cases `measure` by name. With a private backing field the
  name it sees becomes `.measure`, so that branch has to be renamed or it silently stops
  deep-cloning the measure.

### 1.2 `output_dim` cannot be reset to "infer again"

`task$output_dim = NULL` errors, so once set there is no supported way back to inference.

*Suggestion:* `assert_int(rhs, lower = 1L, null.ok = TRUE, coerce = TRUE)` in the setter. One line,
plus a test asserting that assigning `NULL` restores the inferred value.

### 1.3 A `target_batchgetter` with `...` never receives `x`

`R/task_dataset.R` tests `"x" %in% formalArgs(f)`, so `function(data, ...)` silently never gets the
feature tensors.

*Suggestion:* prefer erroring over silently supporting it. Passing `x` into `...` would make the
behaviour depend on whether the function happens to tolerate an unused argument, which is worse than
a clear message. In `TaskTorch$initialize()` (and the `LearnerTorchModule` argument), reject a
batchgetter that has `...` but no `x` with a message pointing at the `x` argument. If you would
rather support it, the change is `"x" %in% formalArgs(f) || "..." %in% formalArgs(f)` in
`task_dataset()`, but then document that `...` receives `x`.

### 1.4 Duplicate target names are accepted

`as_task_torch(d, target = c("y", "y"))` constructs, then fails with a confusing "cannot infer the
output dimension" message because the `length(target) == 1L` branch is skipped.

*Suggestion:* `assert_character(target, any.missing = FALSE, unique = TRUE, null.ok = TRUE)` in the
constructor.

## 2. Design questions worth a second opinion

### 2.1 `predict_types` defaults to `c("response", "prob")` for every `"torch"` learner

The task is unknown at learner construction, so the learner cannot know whether probabilities are
available. On a task whose encoder does not produce them, `predict_type = "prob"` yields a
response-only prediction; `mlr3` warns at score time, and the inferred numeric case now errors up
front, but a custom `prediction_encoder` that omits `prob` still degrades silently.

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

### 2.2 `hash_input.function()` ignores the closure environment

It hashes formals plus body text, so two batchgetters with identical bodies but different captured
values hash alike. Left consistent with how `mlr3torch` already hashes module generators, but it is a
real hole in task hashing: two `TaskTorch` objects that differ only in a captured value collide, and
`benchmark()` would keep one.

*Suggestion:* do not special-case this in `TaskTorch` — fix or document it at the `hash_input()`
level so module generators, batchgetters and encoders behave the same. A cheap improvement is to
additionally hash the values of the variables the function's body actually references from its
enclosing environment (`codetools::findGlobals(f, merge = FALSE)$variables` intersected with
`ls(environment(f))`), falling back to the current behaviour for anything unserialisable. If that is
too fragile, document the limitation in `?TaskTorch` under the `$hash` field, since a silent
benchmark collision is the kind of thing users should be warned about.

### 2.3 Should `as_task_torch()` accept an existing `Task`?

Converting a `TaskClassif`/`TaskRegr` into a `TaskTorch` to attach a custom encoder is a plausible
want, and `as_task_regr()` and friends have `Task` methods.

*Suggestion:* add `as_task_torch.Task()` that takes `x$backend` and then copies `col_roles` and
`row_roles` wholesale, rather than re-deriving the target from an argument. The traps are that
`col_roles` carries `stratum`, `group`, `offset` and the two weight roles — which are all legal for
`"torch"` because its `task_col_roles` entry is copied from `regr` — and that the source task's
`$filter()`/`$select()` state lives in the roles, not the backend, so copying the roles is what
preserves it. Worth a test that a filtered, stratified source task survives the round trip.

### 2.4 No dictionary entry per problem

`lrn("torch.module")` and `po("torch_model")` are shared by everyone, so a custom problem cannot be
handed to a colleague as a learner id.

*Suggestion:* this is inherent to the one-task-type design and should not be fixed in the package.
Document the recipe instead: users can `mlr_learners$add("my.problem", function(...) ...)` with a
constructor that fills in their `module_generator`, `ingress_tokens` and `loss`. A short subsection
in the *Custom Learning Problems* article would cover it.

## 3. Housekeeping

### 3.1 Decide the fate of the full task-type walkthrough

`attic/custom_task_type_full.Rmd` is the "do it properly" article that the `TaskTorch`-only rewrite
displaced: multi-label built from scratch as a real task type, plus the VAE task type.

**It is a reconstruction.** The rendered original was lost when the session scratchpad was cleaned,
so this copy has not been re-rendered. Diff it against your expectations and re-render before
trusting it.

*Suggestion:* promote it back to `vignettes/articles/custom_task_type_full.Rmd` with the title
"Adding a Custom Task Type", add it to the `tutorials` menu in `_pkgdown.yml` under the quick
article, and cross-link the two: the quick one already has a "What You Give Up" section that is the
natural place to link from. The material is written and was working; dropping it loses the only
end-to-end account of the real extension path. If you would rather not maintain two articles, delete
the file rather than leaving it in `attic/` to rot.

### 3.2 Remove `TODOs.md` before merging

The file and its `.Rbuildignore` entry (`^TODOs\.md$`) should both go once these items are resolved
or moved to issues.

## 4. Known-good, do not re-investigate

Recorded so nobody spends time on these again:

* **6 failing tests are pre-existing**, reproduced on a pristine checkout of the branch commit:
  4 × `test_LearnerTorchTabM.R` man-page lookups, and `test_nn_graph.R:209` /
  `test_PipeOpTorchBlock.R:37` deep-clone hash mismatches (`hash_input.nn_module()` uses
  `data.table::address()`, which a deep clone cannot preserve).
* **`future::multisession` appears broken under `pkgload::load_all()`** with
  `could not find function ".__LearnerTorchModule__.dataset"`. This is a `load_all` artifact:
  `leanify_package()` moves R6 methods into the namespace and workers load the *installed*
  `mlr3torch`, which lacks any method added since. Verified working against an installed build.
  Any newly added R6 method will look broken this way until reinstalled.
* **`lrn("torch.module")`'s dictionary prototype needs a dummy `loss`** (`R/LearnerTorchModule.R`),
  because there is no default loss for the `"torch"` task type and `as.data.table(mlr_learners)`
  builds prototypes. It is not a default at `lrn()` time — that still errors, by design.
* **A semantically mismatched learner and task runs silently.** This is the documented cost of one
  shared task type, not a defect.
* **Preprocessing works.** Generic `mlr3pipelines` ops (`scale`, `pca`, `encode`, `removeconstants`,
  `imputemean`, `colapply`) and `mlr3torch`'s `lazy_tensor` ops (`trafo_*`, `augment_*`) all train on
  a `TaskTorch`, return a `TaskTorch`, and work inside pipelines and under `resample()`. Ops that
  interpret the target (`classbalancing`, `smote`, `ppl("targettrafo")`) correctly reject it.
