# mlr3torch — open TODOs

Carried over from the audit of `main` @ `736f5165` (8 agents: adversarial bug hunt, doc/code
correctness, empirical usage sweep of 135 configuration checks, doc coverage). Everything the audit
listed that has since been fixed on `main` is dropped from this file; what is left is open.

**[D]** marks items that need a decision, not just a patch.

Unmerged work that closes items listed here:

| Where | Covers |
|---|---|
| PR #499 (`more-fixes`) | 1.1, 1.4, 1.9h, 1.10, 2.1, 2.2, 2.3, the scaling half of 5.0 |
| PR #496 (`feat/best-valid-scores`) | 1.3, as an opt-in `$best_valid_scores` |
| branch `audit-fixes` | 1.0 |
| branch `fix/history-requires-measures` | 1.9g |
| branches in the `torch` checkout at `/Users/sebi/mlr/torch` | the upstream halves of 1.2 and 1.9b |

---

## 1. Bugs — code

### 1.0 `rr$learners[[i]]` is silently the wrong learner

`resample(..., store_models = TRUE)$learners` comes back in hash order rather than iteration order
for learners holding an `nn_module` hyperparameter. `$score()` and `as.data.table(rr)` are not
affected; the damage is confined to `rr$learners` / `bmr$learners`.

```r
mk = function() lrn("classif.mlp", epochs = 1, batch_size = 50, neurons = 5, activation = nn_relu)
res = rsmp("cv", folds = 3); res$instantiate(tsk("iris"))
rr = resample(tsk("iris"), mk(), res, store_models = TRUE)
# rr$learners[[3]] is iteration 1's model
```

**Cause:** `Learner$hash` is `calculate_hash(class, id, param_set$values, ...)`, and
`calculate_hash()` applies `hash_input()` only to each top-level argument. `param_set$values` is a
plain list and `mlr3misc` has no `hash_input.list`, so `digest()` serializes the list wholesale --
including the `nn_relu` closure *together with its environment*, which torch mutates when the module
is first instantiated. Each iteration therefore records a different `learner_hash`, and
`ResultData$learners()` merges on it with `sort = TRUE`.

**Fix** (implemented on `audit-fixes`, never merged): `hash_input.list()`
(`function(x, ...) map(x, hash_input)`) makes the element-wise methods reachable, and
`hash_input.nn_module()` keys on the module class plus the public methods of its R6 generator
instead of `data.table::address()`. Both in `R/LearnerTorch.R`; regression tests in
`test_LearnerTorch.R`.

**Note for review:** `hash_input.list` is an S3 method on another package's generic for a base type,
so it takes effect for *every* package in a session that loads mlr3torch. It is semantically a no-op
for lists of plain data, but the natural home for it is `mlr3misc`. Worth proposing upstream and
dropping here once it lands.

### 1.2 `cross_entropy` `ignore_index` off by one — upstream, waiting for a release

mlr3torch label-encodes multiclass targets **1-based**, `ignore_index` is forwarded to torch
unchanged, and torch reads it 0-based, so `ignore_index = k` ignores class `k + 1`.

Root cause is a `torch` bug: `torch_cross_entropy_loss()` converts `target` via `to_index_tensor()`
but forwards `ignore_index` unconverted, even though libtorch compares it against the already
converted target. Same omission in `torch_nll_loss()`, `torch_nll_loss2d()`, `torch_nll_loss_nd()`.
**Fixed upstream in torch**; nothing to do here beyond raising the `torch` version requirement in
`DESCRIPTION` once the fix is released. Deliberately not worked around in mlr3torch -- a shift here
would have to be removed again. Documented in a comment at `R/TorchLoss.R:308`.

**Still open regardless:** nothing validates `ignore_index` against `seq_along(task$class_names)`,
so an out-of-range value silently ignores nothing:
```r
t_loss("cross_entropy", ignore_index = 17L)$generate(tsk("iris"))   # accepted
```

### 1.3 [D] `internal_valid_scores` reports the last epoch, `internal_tuned_values` the best one
`R/learner_torch_methods.R:226` vs `R/LearnerTorch.R:470`.

After early stopping these describe **different models**, so a tuner archives the wrong
configuration's performance.

```r
l = lrn("regr.mlp", epochs = 30, batch_size = 8, neurons = c(200, 200), p = 0,
  patience = 3, validate = 0.3, measures_valid = msr("regr.mse"), seed = 3,
  opt.lr = 0.5, callbacks = t_clbk("history"))
l$train(tsk("mtcars"))
# internal_tuned_values$epochs = 5 (MSE 216), internal_valid_scores = 378 (epoch 8)
```

Why this is a bug rather than a convention:

- `mlr3learners`' xgboost indexes its evaluation log at
  `attributes(model)$early_stop$best_iteration`, i.e. exactly the iteration reported in
  `internal_tuned_values`.
- `tnr("internal")` + `msr("internal_valid_score")` ranks external configurations by this number, so
  it ranks them by the score of models that are then discarded. A **reproducible rank flip** was
  demonstrated on sonar (seed 4, `patience = 2`): reported scores pick `lr = 0.005`, but at each
  configuration's own tuned epoch `lr = 0.02` is better (0.2097 vs 0.2258).
- With `classif.ft_transformer` the gap reached an order of magnitude.
- It does not even require early stopping to fire.

**Decision needed.** PR #496 adds `$best_valid_scores` plus `msr("best_valid_score")`, i.e. option
(b) made opt-in; `msr("internal_valid_score")` still reports the last epoch and still silently
misranks. Open: whether the default should change, or at least warn when `internal_valid_score` is
used as the tuning measure with `patience > 0`.

Related: `restore_best_weights` (in `main`, initialized to `FALSE`) is option (a) made opt-in.
Whenever it is set, the stored network *is* the best epoch's, and reporting the last epoch's scores
becomes plainly wrong rather than merely inconsistent -- the two decisions are linked.

### 1.9b Two exposed hyperparameter levels that always failed — fixed upstream in `torch`
`attention_bias = FALSE` on the FT-Transformer, and `anneal_strategy = "linear"` on
`t_clbk("lr_one_cycle")`, both fail unconditionally (verified against torch 0.17.0):

- `nnf_multi_head_attention_forward()` leaves `k` and `v` unassigned when `bias = FALSE` and `query`
  differs from `key` -- mlr3torch always hits this, since the last FT block sets `query_idx = -1`.
  Fails with `object 'k' not found`.
- `lr_one_cycle()` assigns its annealing function to `self.anneal_func` instead of
  `self$anneal_func`, so the linear strategy never sets it. Fails with
  `attempt to apply non-function`.

**Nothing to do in mlr3torch** beyond raising the `torch` requirement in `DESCRIPTION` once the
fixes are released. Until then both remain reachable levels that a tuner samples.

### 1.9g `t_clbk("history")` with no measures silently produces an empty table
```r
l = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 5, callbacks = t_clbk("history"))
l$train(tsk("iris")); l$model$callbacks$history
#> Empty data.table (0 rows and 1 cols): epoch
```
"Saves the training and validation history during training" reads like it will record *something*.
It only ever records `measures_train`/`measures_valid`; the training loss -- the one quantity that
always exists -- is never logged.

A fix exists on `fix/history-requires-measures` (`e678021b`): it errors when neither
`measures_train` nor `measures_valid` is set, as an `Mlr3ErrorConfig` so the fallback learner is not
triggered. Fallout: three tests in `test_LearnerTorch.R` attach the callback incidentally and need a
measure. Not done there: logging the training loss by default, the other option.

### 1.9i [D] `mirai` backend silently produces irreproducible results
```r
daemons(2, .compute = "mlr3_parallelization")
set.seed(42); b = resample(tsk("sonar"), mk(), rsmp("cv", folds = 3))
set.seed(42); c = resample(tsk("sonar"), mk(), rsmp("cv", folds = 3))
identical(gp(b), gp(c))   #> FALSE      (TRUE under future)
```
Root cause is upstream: `mlr3:::future_map`'s mirai branch calls `mirai_map()` without seeding,
while the `future` branch passes `future.seed = TRUE`. `classif.ranger` reproduces it, so it is not
mlr3torch-specific. It is recorded here because **mlr3torch learners are stochastic by default**
(`seed = "random"`), so torch users are maximally exposed, and nothing in the docs suggests pinning
`seed` before going parallel. Setting the learner's `seed` explicitly makes mirai reproducible and
equal to the sequential result. Worth reporting upstream and mentioning in the parallelization docs.

---

## 2. Test-suite gaps

All three gaps the audit found (multiple unknown dimensions in the shape-inference helper, a
checkpoint test writing into a path a previous run used, the `lr_reduce_on_plateau` paths without
validation scores) are closed on PR #499.

---

## 3. Documentation — corrections still open

### 3.3 [D] Two unused bibentries
`arik2021tabnet` and `ioffe2015batch` in `R/bibentries.R` are referenced by no `format_bib()` call.
Dead weight, but deleting a reference is a judgement call -- they may be intended for learners not
yet written up.

### 3.4 "allows to X" (non-native phrasing) — 12 instances
`R/cache.R:4`, `R/PipeOpModule.R:7`, `R/PipeOpTorchBlock.R:14`, `R/DataDescriptor.R:6`,
`R/LearnerTorch.R:25`, `man-roxygen/paramset_torchlearner.R:101`, `README.Rmd:54`, `README.Rmd:114`,
`NEWS.md:108`, `NEWS.md:109`, `NEWS.md:171`, `DESCRIPTION:35`.
Left alone deliberately -- `DESCRIPTION` is CRAN-visible and this is a style call. Suggested:
"allows one to X" / "makes it possible to X".

---

## 4. Documentation — missing entirely

Ordered by user impact. These are content tasks, none of them started.

### 4.1 `jit_trace` is documented nowhere
Defined at `R/paramset_torchlearner.R:125`, present on every non-`LearnerTorchModule` learner and on
`po("torch_model_*")`, headlined in NEWS 0.2.0 ("can lead to significant speedups") -- yet
`grep -rn "jit_trace" man/ man-roxygen/` returns **zero hits**, and it appears in no vignette or
README. Should live in `man-roxygen/paramset_torchlearner.R` plus the *Important Runtime
Considerations* section of `R/LearnerTorch.R`. Needs: that it runs `torch::jit_trace()` once at the
start of training; that it removes per-batch R interpreter overhead; that it requires
data-independent control flow (no data-dependent branching, no varying shapes); its batch-norm
history (fixed in 0.2.1); and the relation to `jittable`.

### 4.2 Marshaling is reference-only, and is a silent-data-loss footgun
`?LearnerTorch` has a four-line *Saving a Learner* section; that is the only place in the doc set.
`marshal`, `unmarshal`, `saveRDS`, `future`, `mirai` appear in **zero** README or vignette lines,
though `future`/`mirai`/`callr` are all in `Suggests`.

Confirmed empirically: `saveRDS()` of a trained but unmarshaled learner **succeeds**, then
predicting after `readRDS()` fails with `external pointer is not valid`.

Should cover: why torch external pointers don't survive serialization; the `$marshal()`/
`$unmarshal()` round-trip with a worked `saveRDS`/`readRDS` example; that `resample()`/`benchmark()`
with `future`/`mirai` marshal automatically via mlr3's `marshal` property, so the manual call is
only for hand-rolled `saveRDS`; and that `$model$network` is a live torch object that must not be
shared across processes.

### 4.3 No prose on validation, early stopping or internal tuning
`patience`, `min_delta`, `internal_valid_scores`, `$internal_tuned_values`,
`to_tune(..., internal = TRUE)`, `AutoTuner`, `set_validate()` with `"test"`/`"predefined"` --
**zero hits** across `README.Rmd` and all vignettes, despite the README bullet promising that
networks "can be easily tuned via `mlr3tuning` and friends". There is no worked `resample()`,
`benchmark()` or `AutoTuner` example anywhere.

`CallbackSetEarlyStopping` (`R/CallbackSetEarlyStopping.R`) is **not exported, not in
`mlr3torch_callbacks`, and has no man page** -- early stopping is only reachable through the
`patience`/`min_delta` hyperparameters, which is stated nowhere a user would look.

Proposed: a new `vignettes/articles/tuning.Rmd` -- set validation three ways → read
`internal_valid_scores` → configure `patience`/`min_delta` and show `$model$epochs` shrinking →
`epochs = to_tune(upper = 100, internal = TRUE)` combined with an external search space in an
`AutoTuner`, and why that is cheaper than tuning `epochs` externally.

### 4.3b Multi-modal graphs need a `po("select")` per ingress, shown nowhere
`gunion(list(po("torch_ingress_num"), po("torch_ingress_categ"), po("torch_ingress_ltnsr")))` on a
task with all three feature types does not work -- each ingress rejects a task containing any other
feature type. The error is actionable (`Consider using po("select")`) and the
`po("select", selector = selector_type(...))` version trains fine, but no vignette shows the
multi-modal pattern, and it is the natural thing to reach for after reading about the three ingress
flavours.

Related gap: `$shapes_out()` exists only on individual `PipeOp`s, not on a `Graph`
(`g$shapes_out(...)` gives `attempt to apply non-function`). The docs correctly scope it to a single
`PipeOp`, so this is a missing convenience rather than a contradiction -- but it is the obvious
thing to want when debugging a long chain.

### 4.4 `augment_` vs `trafo_` convention is explained once, in the wrong place
The convention carries real behaviour: `?mlr_pipeops_preproc_torch` states that `stages` "is set to
`"train"` when the `PipeOp`'s id starts with `"augment_"` and to `"both"` otherwise". That sentence
is the only statement of it, and it lives on the base-class page. In the vignettes, `stages` gets
one paragraph inside `lazy_tensor.Rmd`'s *Custom Preprocessing* section -- an authoring tutorial,
not a usage one -- and never lists all three levels.

Also: the 23 `trafo_*`/`augment_*` man pages have **no `\description{}` at all**. At minimum each
needs a one-line description plus `@seealso` to the corresponding `torchvision::transform_*`.

### 4.5 Runtime/performance knobs are a parameter list, not guidance
`num_threads`, `num_interop_threads`, `tensor_dataset`, `num_workers`, `pin_memory`,
`worker_globals`, `worker_packages`, `worker_init_fn`, `collate_fn`, `timeout`, `sampler`,
`batch_sampler`, `batch_size_predict`, `shuffle`, `device`/`auto_device` -- all defined in
`man-roxygen/paramset_torchlearner.R` and rendered only into `?LearnerTorch`'s parameter list.
**Zero** occurrences in README or any vignette.

`device = "cuda"` appears once (`pipeop_torch.Rmd:147`) and is silently overridden back to `"cpu"`
in a hidden chunk, so no reader ever sees a GPU discussion. `auto_device()` is exported with a
27-character description and never mentioned in prose. `?LearnerTorch`'s *Important Runtime
Considerations* is four bullets.

### 4.6 Authoring custom losses, optimizers and LR schedulers is undocumented
`get_started.Rmd` covers only *consuming* `t_loss()`/`t_opt()`. `TorchLoss$new()`, `as_torch_loss()`,
`TorchOptimizer$new()`, `as_torch_optimizer()`, `as_lr_scheduler()` appear in **no** vignette or
README line -- even though six of the eleven built-in callbacks are LR schedulers. Working examples
exist only in `@examples`.

Also undocumented in prose: the `loss.` / `opt.` / `cb.<id>.` parameter-prefix convention (currently
one clause at `man/mlr_learners_torch.Rd:116`), and `task_types` filtering.

Proposed: extend `vignettes/articles/callbacks.Rmd` into "Customizing Training", also covering
`callback_set()` vs `torch_callback()` and terminating training via `ctx$terminate` (the *Terminate
Training* section in `R/CallbackSet.R` has no vignette counterpart).

### 4.7 Writing a custom `PipeOpTorch` has a reference contract but no tutorial
`?mlr_pipeops_torch` has a solid *Inheriting* section, but no vignette ever shows
`R6Class(inherit = PipeOpTorch)`; `internals_pipeop_torch.Rmd` covers `ModelDescriptor` from the
consumer side only.

Meanwhile the dev version exports 13 new shape helpers (`infer_shapes`, `broadcast_shapes`,
`resolve_dim`, `shape_to_str`, `assert_shape`, `assert_shapes`, `assert_known_dims`, `assert_ndim`,
`assert_same_ndim`, `assert_same_batch_size`, `assert_dim_in_range`, `assert_not_batch_dim`,
`assert_positive_extent`, `reshape_output_shape`) as bare reference stubs, and **removes the
`only_batch_unknown` constructor argument** -- a breaking change for exactly the audience that has
no tutorial, with no migration note.

Proposed: `vignettes/articles/custom_pipeop_torch.Rmd` under the existing "Internals" menu group --
implement one non-trivial layer end to end, show `.shapes_out()` handling `NA` in *any* dimension
via `assert_known_dims()`, show registration so `nn("mykey")` works, and add the
`only_batch_unknown` migration note.

### 4.8 Vignettes don't ship with the package
`.Rbuildignore` contains `^vignettes/articles$` and `DESCRIPTION` has **no** `VignetteBuilder`
field, so the installed/CRAN package has no vignettes: `browseVignettes("mlr3torch")` is empty and
`?mlr3torch` offers no entry point.

Compounding this, **no man page in `man/` contains a `vignette(` call or an `articles/` link** --
the reference docs and the website tutorials are two disconnected islands. The README's only pointer
is the unlinked sentence "Start by reading one of the vignettes on the package website!"
(`README.Rmd:211`).

Fix: either add `VignetteBuilder: knitr` and un-ignore `vignettes/articles`, or accept website-only
but add real article URLs to the README and `@seealso` links from `?LearnerTorch`, `?lazy_tensor`,
`?mlr_pipeops_torch`, `?TorchCallback`.

### 4.9 Caching and downloads
- `options(mlr3torch.cifar_download_timeout = <seconds>)` is read at `R/TaskClassif_cifar.R:60`
  (default 5400s) and exists **only as a source comment** -- no man page, no `@section Options`.
- `mlr3torch.cache` is documented in `?mlr3torch-package` but has **zero** hits in README and
  vignettes, even though `lazy_tensor.Rmd` and `pipeop_torch.Rmd` silently trigger
  MNIST/tiny-imagenet downloads (`pipeop_torch.Rmd` hides the download in an `include = FALSE`
  chunk).
- No task man page states its download size.
- The cache-version invalidation mechanism (`CACHE$versions`, `version.json`, and the error
  "Cache directory '%s' was not initialized by mlr3torch" at `R/cache.R:60`) is `@noRd`, so a user
  who hits that error has **no documented recovery**.

### 4.10 No fine-tuning / transfer-learning documentation
Despite 53 pretrained vision learners. `vision_learner_list.Rmd` is a bare table; `pretrained` gets
one sentence on `?mlr_learners.torchvision`. `replace_head()`'s description is a duplicated sentence
fragment ("Replace the head of a network Replace the head of the network with a linear layer with
d_out classes."). `CallbackSetUnfreeze` / `t_clbk("unfreeze")` -- the intended companion for staged
fine-tuning -- appears in no vignette. Nothing documents where pretrained weights are downloaded to,
how large they are, or the input-size/normalization requirements per architecture.

### 4.11 `seed` semantics are misleadingly documented
Both README and `get_started.Rmd` demonstrate reproducibility with plain `set.seed()`. The actual
`seed` hyperparameter -- `integer(1)` | `"random"` | `NULL`, initialised to `"random"`, stored in
`$model$seed`, **with the consequence that clones of a learner use a different seed** -- is
documented only in `man-roxygen/paramset_torchlearner.R:21-28` and appears in no narrative doc.

### 4.12 ~115 hyperparameters are settable but not lookupable
Man pages with **no Parameters section at all**:

| object | undocumented ids |
|---|---|
| `po("nn_ft_transformer_block")` | `attention_n_heads`, `attention_dropout`, `attention_initialization`, `attention_normalization`, `attention_bias`, `ffn_d_hidden`, `ffn_d_hidden_multiplier`, `ffn_dropout`, `ffn_activation`, `ffn_normalization`, `ffn_bias_first`, `ffn_bias_second`, `residual_dropout`, `prenormalization`, `is_first_layer`, `query_idx` (16) |
| `po("nn_ft_cls")` | `initialization` |
| `t_clbk("lr_one_cycle")` | `max_lr`, `total_steps`, `pct_start`, `anneal_strategy`, `cycle_momentum`, `base_momentum`, `max_momentum`, `div_factor`, `final_div_factor`, `verbose` (10) |
| `t_clbk("lr_reduce_on_plateau")` | `mode`, `factor`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`, `eps`, `verbose` (8) |
| `t_clbk("lr_cosine_annealing")` | `T_max`, `eta_min`, `last_epoch`, `verbose` |
| `t_clbk("lr_lambda")`, `t_clbk("lr_multiplicative")` | `lr_lambda`, `last_epoch`, `verbose` |
| `t_clbk("lr_step")` | `step_size`, `gamma`, `last_epoch` |
| all 23 `trafo_*`/`augment_*` | 2–8 each, incl. `stages`/`affect_columns` on every one (~100 total) |
| all learners + `po("torch_model_*")` | `jit_trace` (see 4.1) |

For the preprocessing family most of this could be fixed mechanically with a per-argument gloss plus
`@seealso torchvision::transform_*`.

### 4.13 Learner help topics don't follow mlr3 naming
Topics are `mlr_learners.mlp`, not the mlr3-conventional `mlr_learners_classif.mlp`, so
`?mlr_learners_classif.mlp` **fails for all 60 learners** (`lrn(...)$help()` works, since it uses
`$man`). The 53 torchvision learners share the single topic `mlr_learners.torchvision` with two
aliases, so `?mlr_learners_classif.resnet18` and `?classif.resnet18` both fail.

Similarly, the 4 LR-scheduler callbacks (`lr_cosine_annealing`, `lr_lambda`, `lr_multiplicative`,
`lr_step`) have no `mlr_callback_set.<key>` topic and land on a generic page with no Parameters
section.

### 4.14 Thinly documented topics
Description length in characters:

| topic | chars | description |
|---|---|---|
| `mlr_pipeops_nn_gelu` | 4 | "Gelu" |
| `PipeOpPreprocTorchTrafoNop` | 13 | "Does nothing." |
| `mlr_learners.tab_resnet` | 15 | "Tabular resnet." — a full learner with a paper behind it |
| `lazy_tensor` | 21 | "Create a lazy tensor." — the package's signature data type |
| `mlr_pipeops_nn_rrelu` | 22 | |
| `auto_device` | 27 | "First tries cuda, then cpu." |
| `mlr_pipeops_nn_flatten` | 27 | says nothing about what it flattens |
| `as_lazy_tensor` | 34 | conversion semantics from `dataset`/`torch_tensor`/`numeric` not distinguished |
| `as_torch_loss` / `as_torch_optimizer` / `as_torch_callback` | 34–39 | the three main extension entry points |
| `mlr_learners.module` | 43 | the primary custom-architecture entry point |
| `mlr_pipeops_nn_fn` | 45 | never mentioned in any vignette either |
| `as_lr_scheduler` | 64 | |
| `replace_head` | — | description is a duplicated sentence fragment |

Plus the 23 preprocessing pages with no `\description{}` (see 4.4).

### 4.15 Suggested new vignettes
- **`practical_guide.Rmd` — "Training in Practice"** (absorbs 4.1, 4.2, 4.5, 4.9, 4.11): device
  selection and `auto_device()`; making training fast (`tensor_dataset`, `batch_size`, threads,
  `jit_trace` and when not to use it); parallel dataloading (`num_workers`, `worker_globals`);
  reproducibility; caching and downloads; saving/loading and parallelization; troubleshooting
  (shape-mismatch errors, `Mlr3ErrorConfig`, graph ID clashes, OOM).
- **`tuning.Rmd` — "Validation, Early Stopping and Tuning"** (4.3).
- **`vision.Rmd` — "Images and Transfer Learning"** (4.4, 4.10): loading images from a folder into a
  `lazy_tensor` task -- the motivating use case `lazy_tensor.Rmd` asserts but never demonstrates --
  plus a multi-modal `tsk("melanoma")` example (tabular + `lazy_tensor` in one task, a
  README-advertised feature with no example anywhere).
- **`custom_pipeop_torch.Rmd` — "Writing your own PipeOpTorch"** (4.7).
- Extend **`callbacks.Rmd` → "Customizing Training"** (4.6).

---

## 5. UX / polish

### 5.0 [D] The untuned MLP is weak and high variance
```r
benchmark(benchmark_grid(tsk("iris"),
  list(lrn("classif.mlp", epochs = 20, batch_size = 32, neurons = 10, device = "cpu"),
       lrn("classif.featureless")), rsmp("cv", folds = 2)))$aggregate(msr("classif.ce"))
```

**The original dropout diagnosis did not hold up.** Mean CE over 5 seeds, iris, 2-fold CV:

| `p` | 0 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| CE | 0.539 | 0.525 | 0.501 | 0.507 |

featureless scores 0.667 on the same setup, i.e. **every** dropout level beats it on average, and
the differences between them are within noise. `p` has since been changed to `0.1` anyway, for
consistency with the other learners -- not as a fix for this.

What remains open is whether any further defaults should change; the likely levers are learning
rate, `epochs` and feature scaling. (The scaling half is addressed on PR #499, which adds a shared
*Input Scaling* section to the tabular learners.)

### 5.1 No hint to marshal on `external pointer is not valid`
`saveRDS()` of an unmarshaled trained learner succeeds; predicting after `readRDS()` then fails with
`external pointer is not valid`. Technically correct, and the documented flow is marshal-first, but
a targeted message ("did you forget `$marshal()`?") would save real time. See also 4.2.

### 5.2 `t_clbk("early_stopping")` gives an unhelpful error
`Element with key 'early_stopping' not found in DictionaryMlr3torchCallbacks!` -- without listing
valid keys or mentioning that early stopping is configured via the learner's `patience`/`min_delta`
parameters rather than a dictionary callback. A plausible thing for a user to get wrong.

### 5.3 Predicting on a marshaled learner silently works
`$marshaled` stays `TRUE` and `model$network` is `NULL`, yet prediction returns bit-identical
results (mlr3 unmarshals internally). Harmless, but reads as inconsistent.

### 5.4 [D] A hand-built graph and `lrn("classif.mlp")` don't match at equal seed
Same architecture, same seed, same optimizer/loss/batch_size/epochs. Both networks are structurally
identical (83 params, shapes `10x4 | 10 | 3x10 | 3`), yet **initial weights already differ at
`epochs = 0`**, so predictions diverge (max abs diff 0.18 with `shuffle = FALSE`).

Reproducibility *within* each construction path is exact (same seed → bit-identical, different seed
→ different). What fails is equality *across* construction paths -- presumably the torch RNG is
seeded at a different point relative to module construction in `LearnerTorchMLP` vs the
`PipeOpTorch`/`ModelDescriptor` path.

Worth a decision because "does my hand-built graph reproduce the built-in learner?" is the natural
way a user validates a custom architecture, and it silently answers no.

Minor related detail: `classif.mlp` always inserts an `nn_dropout` module (param names `0`, `3` ⇒
slots 0..3) even at `p = 0`.

---

## Appendix A — investigated and rejected

Recorded so they are not re-reported as findings.

- **`t_loss("cross_entropy", ignore_index = NULL)` erroring is correct.** `ignore_index` is a
  `p_int`, so paradox rejects `NULL` with a message that names the parameter and the expected type.
  This is unlike `class_weight`, a `p_uty` whose documented default *is* `NULL` -- that case was a
  real bug and is fixed.

- **`nn_squeeze` squeezing the batch dimension is intended, not a bug.** An audit agent flagged that
  `po("nn_squeeze", dim = 1)$shapes_out(list(c(1, 4, 3)))` returns `c(4, 3)` while every sibling
  operator that changes a dimension guards with `assert_not_batch_dim()`. Adding that guard breaks
  `tests/testthat/test_PipeOpTorchReshape.R:146`, which asserts the current behaviour *and* verifies
  on line 149 that the module itself does the same thing at runtime. The shape inference therefore
  correctly mirrors torch. The `assert_not_batch_dim()` example in `R/shape.R:450` happens to use
  `id = "nn_squeeze"`, which is misleading -- it is illustrative only.

- **The default optimizer is Adam, and the code docs saying so are correct.** Only the historical
  0.2.0 NEWS entry was wrong, and it has been annotated.

---

## Appendix B — verified clean, don't re-audit

**Docs**: all 124 runnable examples pass (extracted via `tools::Rd2ex` with `\dontrun`/`\donttest`
uncommented, each run in a fresh session); all 6387 cross-references resolve; `tools::checkDocFiles()`
clean; 142 R6 method blocks have `@param` == formals in both directions; no `@field` documents a
non-existent field; ParamSet ids match prose everywhere except the cases listed above; all `@inherit`
targets exist; `man/` is byte-identical to a fresh `roxygenise()`.

**Code**: marshal/unmarshal round-trips bit-identical (including `jit_trace = TRUE` /
`script_module`); row ordering safe (`.dataloader_predict()` forces `shuffle = FALSE`,
`drop_last = FALSE` and drops `sampler`/`batch_sampler`; batched predictions on a filtered,
reordered task match per-row predictions; `tensor_dataset = TRUE` matches `FALSE`); `network$eval()`
correctly set at predict time; `pointer_shape` vs `pointer_shape_predict` guard correctly rejects
train-only augmentation; seed sampled once and reused at predict; `clone(deep = TRUE)` fully
independent; `lazy_tensor` columns sharing a dataset materialize independently; shape inference
correct under all 2^n unknown-dimension patterns (now swept by the test suite, see PR #499).

**Empirical**: 135 configuration checks across 8 areas, 131 passed. All 8 learner types on
binary/multiclass/regression (probs sum to 1, column order matches `class_names`,
`response == argmax(prob)`, factor levels preserved incl. classes absent from training); every
documented parameter demonstrably takes effect; graph building incl. merges, blocks, `nn_fn`,
reshape round-trips; `materialize`/`DataDescriptor`; `trafo_normalize` exact and
train/predict-identical; `augment_hflip` correctly train-only; all callbacks; all four `validate`
modes; `resample`/`benchmark`/`AutoTuner`/`tnr("internal")`; `po("learner")` in graphs;
`GraphLearner` marshal round-trip; callr encapsulation; all 17 edge cases (single feature,
single-row predict, constant feature, `batch_size` > n, `predict_newdata`, clear errors for
factor-only features and missing values).

---

## Appendix C — work in flight

Branches that exist but are not merged, so that none of it gets lost. All of them live in a
worktree under `.claude/worktrees/` and none of them is pushed yet.

| Worktree | Branch | Addresses | Status |
|---|---|---|---|
| `more-fixes` (the main checkout) | `more-fixes` | 1.1, 1.4, 1.9h, 1.10, 2.1, 2.2, 2.3, scaling docs of 5.0 | PR #499, open |
| `pipeop-torch-helper` | `feat/pipeop-torch-helper-fn` | issues #144, #403, #398 | committed, not pushed |
| `predict-progress` | `feat/predict-progress` | issue #435 | committed, not pushed |
| `model-printer` | `feat/learner-torch-model-printer` | issue #393 | committed, not pushed |
| `torch-model-dict` | `feat/torch-model-dictionary` | issue #376 | committed, not pushed |
| `nn-examples` | `docs-nn-in-examples` | issue #346 | committed, not pushed |

`feat/pipeop-torch-helper-fn` adds `pipeop_torch()`, which generates the `PipeOpTorch` R6 class
from an `nn_module` (`auxiliary` names the module arguments that follow from the input shape,
output shapes are traced on the meta device unless given), an `as_pipeop()` method for
`nn_module_generator`s, and the article *Writing your own PipeOpTorch* -- which is 4.7 of this
file. The custom-`PipeOpTorch` example moved out of `?mlr_pipeops_torch` into that article.

`feat/predict-progress` gives the prediction loop its own callback stages (`predict_begin`,
`predict_batch_end`, `predict_end`) with a `ContextTorchPredict`, and `t_clbk("progress")` uses
them. Only callbacks implementing one of the three are constructed at prediction time, so the
checkpoint callback does not create a directory when a learner predicts.

`feat/learner-torch-model-printer` adds `print.learner_torch_model()`, which used to dump the
optimizer state dict.

`feat/torch-model-dictionary` only adds a regression test: `classif.torch_model` and
`regr.torch_model` have been in the dictionary since #117, so **issue #376 can be closed as already
done**.

`docs-nn-in-examples` converts `po("nn_x")` to `nn("x")` in the man-page examples, the vignettes and
the README. Package code is deliberately untouched, because those ids become the module names of
the network. Note this branch and `feat/pipeop-torch-helper-fn` both rewrite the examples of
`?mlr_pipeops_torch`, so expect a conflict when merging the second one.

The branch is `docs-nn-in-examples`, not `docs/nn-in-examples`: the remote has a branch named
`docs`, i.e. the file `refs/heads/docs`, so no ref can live below that name.

Note that every branch off `main` still carries the `_pkgdown.yml` bug that breaks the pkgdown
build (`equals-.lazy_tensor` has to be `` "`==.lazy_tensor`" ``); it is fixed on `more-fixes`.
