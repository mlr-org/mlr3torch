# mlr3torch — open TODOs from the audit

Everything here is **still open**. Items already fixed on branch `audit-fixes` are listed in `NEWS.md`
and are not repeated below.

Source: 4-agent audit of `main` @ `736f5165` — adversarial bug hunt, doc/code correctness,
empirical usage sweep (135 configuration checks), doc coverage.

**[D]** marks items that need a decision from you, not just a patch.

---

## 1. Bugs — code

### 1.1 [D] Checkpoint callback rejects any pre-existing directory
`R/CallbackSetCheckpoint.R:43` — `assert_path_for_output(path)` without `overwrite = TRUE`.

Makes checkpointing **unusable with `resample()`, `benchmark()` and tuning** — it fails on the second
iteration. Also blocks training the same learner twice and pointing at a directory you created yourself.

```r
d = tempfile()
lx = lrn("classif.mlp", epochs = 1, batch_size = 32, neurons = 5,
  callbacks = t_clbk("checkpoint", path = d, freq = 1))
resample(tsk("iris"), lx, rsmp("cv", folds = 2))
#> Error: Assertion on 'path' failed: File at path already exists
```

*Why this needs a decision:* `tests/testthat/test_CallbackSetCheckpoint.R:38` explicitly asserts the
error ("error when using existing directory"), so the guard is deliberate — but the immediately
following `if (!dir.exists(path)) dir.create(path, recursive = TRUE)` is then dead code, so the
intent is genuinely ambiguous. Options:
- (a) `overwrite = TRUE` and drop the test;
- (b) keep the guard but write into a per-iteration subdirectory so resampling works;
- (c) keep as-is and document the limitation prominently.

### 1.2 [D] `cross_entropy`'s `ignore_index` is off by one
`R/TorchLoss.R:308` (parameter), `:355` (docs).

mlr3torch label-encodes multiclass targets **1-based** (as `?LearnerTorch` "Network Head and Target
Encoding" documents). `ignore_index` is forwarded to torch unchanged, and torch reads it 0-based.

```r
inp = torch_tensor(matrix(c(10,0,0, 0,10,0, 0,0,10), nrow = 3, byrow = TRUE))
for (ii in 0:3) {
  lf = t_loss("cross_entropy", ignore_index = ii)$generate(tsk("iris"))
  cat(ii, sapply(1:3, function(t)
    as.numeric(lf(inp, torch_tensor(rep(t, 3L), dtype = torch_long())))), "\n")
}
# ignore_index=0 -> NaN ...   ignores class 1
# ignore_index=1 -> ... NaN   ignores class 2
# ignore_index=3 -> no NaN    ignores nothing
```

Fix would be `args$ignore_index = args$ignore_index - 1L` plus
`assert_choice(ignore_index, seq_along(task$class_names))`. **Decision needed** because it changes
user-visible semantics: shift to match the documented 1-based encoding, or keep 0-based and document it.

### 1.3 `nn_squeeze` can squeeze away the batch dimension
`R/PipeOpTorchReshape.R:88-100`.

Every sibling operator that changes a dimension guards with `assert_not_batch_dim()` (`nn_glu`
`R/PipeOpTorchActivation.R:789`, `nn_reglu` `:861`, `nn_geglu` `:933`). `nn_squeeze` does not.

```r
po("nn_squeeze", dim = 1)$shapes_out(list(c(1, 4, 3)))
#> $output
#> [1] 4 3      # batch dimension gone; expected an error
```

Reachable through the documented API since `21aa514a` allowed a known batch size
(`unknown_batch = NULL`, `R/ModelDescriptor.R:72`), and at runtime for a trailing batch of size 1.
Fix: call `assert_not_batch_dim(true_dim, shape, self$id)` in `.shapes_out()` and filter dim 1 out of
`.shape_dependent_params()`.

### 1.4 Conv/pool shape inference returns `double`, not `integer`
`R/PipeOpTorchAvgPool.R:68`, `R/PipeOpTorchConv.R:166`.

Both compute `floor((...) / stride + 1)`, which yields doubles. `PipeOpTorch$shapes_out()` coerces its
*inputs* with `assert_shapes(..., coerce = TRUE)` (`R/PipeOpTorch.R:338`) but never its *outputs*, and
`.train()` assigns the result straight into `pointer_shape`. `DataDescriptor$new()` and
`transform_lazy_tensor()` call `assert_shape()` without capturing the coerced return
(`R/DataDescriptor.R:106`, `R/lazy_tensor.R:276`), so the doubles persist and make `identical()`/hash
comparisons of otherwise-equal shapes disagree.

```r
typeof(po("nn_max_pool2d", kernel_size = 2)$shapes_out(list(c(NA, 3, 8, 8)))[[1]])
#> "double"    # expected "integer"
```

Suggested fix: coerce centrally in `PipeOpTorch$shapes_out()` —
`set_names(map(private$.shapes_out(...), as.integer), self$output$name)`.

### 1.5 [D] `internal_valid_scores` reports the last epoch, `internal_tuned_values` the best one
`R/learner_torch_methods.R:226` vs `R/LearnerTorch.R:470`.

After early stopping these describe **different models**, so a tuner archives the wrong
configuration's performance.

```r
l = lrn("regr.mlp", epochs = 30, batch_size = 8, neurons = c(200, 200), p = 0,
  patience = 3, validate = 0.3, measures_valid = msr("regr.mse"), seed = 3,
  opt.lr = 0.5, callbacks = t_clbk("history"))
l$train(tsk("mtcars"))
# history MSE: 48790, 285, 22.0, 113, 30.5, 41.4
# internal_tuned_values$epochs = 3  (MSE 22.0)
# internal_valid_scores        = 41.4 (epoch 6)
```

Confirmed end-to-end: `tnr("internal")` + `msr("internal_valid_score")` puts the last-epoch score in
the archive. **Not obviously a bug** — mlr3's `Learner` docs say `.extract_internal_valid_scores()`
returns the "(final)" scores and `mlr3learners`' xgboost behaves identically, so this may be an
intended mlr3-wide convention. Worth deciding explicitly, since the two are reported side by side.

### 1.6 `CallbackSetHistory$load_state_dict()` is never called
`R/CallbackSetHistory.R:55-62`. Nothing in `R/` calls it, so the resume path is dead code — and
`state_dict()` returns `rbind(state, self$prev_state)`, i.e. new epochs *before* older ones, so the
ordering bug is baked in should it ever be wired up. Either finish it or delete it.
*(Static reasoning only.)*

### 1.7 `replace_head.mobilenet_v2` / `.VGG` hardcode the head input size
`R/LearnerTorchVision.R:168` (`nn_linear(1280, d_out)`) and `:188` (`nn_linear(4096, d_out)`), where
every sibling method reads `$in_features`. Correct only for the standard width variants.
*(Static reasoning only — verifying needs a model download.)*

### 1.8 `patience` counts evaluation rounds, not epochs, when `eval_freq > 1`
`CallbackSetEarlyStopping$on_valid_end` only runs on evaluation epochs. Verified: `eval_freq = 3,
patience = 2` stops at epoch 9 after evaluations at 3, 6, 9 — i.e. 6 epochs of stagnation.
Defensible behaviour, but undocumented in the `patience` description.

---

## 2. Test-suite gaps

### 2.1 Shape-inference helper misses multiple unknown dimensions
`tests/testthat/helper_shape_inference.R` only sweeps single-`NA` and batch-plus-one patterns — so
*multiple non-batch unknown dims*, exactly what commit `21aa514a` added, had **no coverage**.

An exhaustive sweep over **all 2ⁿ unknown-dimension patterns** across ~28 `PipeOpTorch`s
(conv/conv_transpose 1–3d, max/avg/adaptive pool incl. `ceil_mode`, flatten/reshape/squeeze/unsqueeze,
linear, head, layer_norm, batch_norm, glu/reglu/geglu, softmax, dropout, ft_cls, identity, merge
sum/prod/cat, multihead_attention) and all 25 registered `trafo_*`/`augment_*` ops found **zero
mismatches** — the feature itself is sound. The gap is in the helper; worth extending so it stays that way.

### 2.2 Checkpoint tests only use fresh `tempfile()` paths
Which is precisely why 1.1 is invisible to CI.

### 2.3 LR-scheduler tests only cover `eval_freq = 1` with validation
`tests/testthat/test_CallbackSetLRScheduler.R:112` — neither `lr_reduce_on_plateau` crash path
(no validation / `eval_freq > 1`) was covered before the fix on this branch.

---

## 3. Documentation — corrections still open

### 3.1 [D] `nn_ft_transformer_block()` documents 7 defaults its signature doesn't have
`R/PipeOpTorchFTTransformerBlock.R`. Real formals (`:55-69`) give defaults only to `ffn_d_hidden`,
`ffn_d_hidden_multiplier`, `query_idx`. The rendered `\usage{}` shows the rest bare, so the man page
contradicts itself:

| line | arg | claim |
|---|---|---|
| `:23` | `ffn_activation` | "Default value is `nn_reglu`" |
| `:29` | `is_first_layer` | "Default value is `FALSE`" |
| `:31` | `attention_normalization` | "Default value is `nn_layer_norm`" |
| `:33` | `ffn_normalization` | "Default value is `nn_layer_norm`" |
| `:40` | `attention_bias` | "Default is `TRUE`" |
| `:42` | `ffn_bias_first` | "Default is `TRUE`" |
| `:44` | `ffn_bias_second` | "Default is `TRUE`" |

The quoted values are the *PipeOp ParamSet inits* (e.g. `:175` `ffn_activation = p_uty(init = nn_reglu)`),
not the function's defaults. Decide: add the defaults to the signature, or reword to "the PipeOp
initialises this to X". (This was the only default-claim mismatch across all 198 man pages.)

### 3.2 `NEWS.md:155` claims a default-optimizer change that never happened
> "The default optimizer is now AdamW instead of Adam." (0.2.0, Breaking Changes)

`R/LearnerTorch.R:253` has been `t_opt("adam")` since commit `acdb57558` (2023-07-14), i.e. before
v0.1.0, and at the v0.2.0 tag `t_opt("adam")` already resolved to `torch::optim_ignite_adam`.
Runtime today: `lrn("classif.mlp")$optimizer$id == "adam"` for mlp, ft_transformer, tab_resnet and tabm.

**The code docs are correct** — `R/LearnerTorch.R:99` ("Defaults to adam.") and
`vignettes/articles/get_started.Rmd:71` ("the Adam optimizer") should **not** be changed. Only the
NEWS entry is wrong. Being a historical entry, probably annotate rather than rewrite.

### 3.3 `nn_avg_pool1d` documents `divisor_override`, which it doesn't have
`R/PipeOpTorchAvgPool.R:99-100`. The parameter is only added when `d >= 2L` (`:20-22`); the doc line
itself even says "Only available for dimension greater than 1." Needs a conditional template, not a
one-line edit.

### 3.4 `nn_prelu` parameter description is incoherent
`R/PipeOpTorchActivation.R:230`: "Number of a to learn. … there is only two values are legitimate".
Needs a rewrite, e.g. "Number of `a` parameters to learn. Although it takes an integer, only two
values are legitimate: …". Same block, `:229`, ends the bullet with `:` instead of `\cr`.

### 3.5 `nn_max_pool*` `kernel_size` type is wrong and malformed
`R/PipeOpTorchMaxPool.R:70`: `` :: (`integer(1))` `` — paren and backtick swapped, *and* `integer(1)`
misdescribes a `p_uty` that accepts vectors.

### 3.6 Whitespace / formatting nits
- `R/PipeOpTorch.R:61` (trailing whitespace, 2-space indent), `:64` (4-space) — continuations
  elsewhere use 3.
- `R/PipeOpTaskPreprocTorch.R:65` — 2-space indent instead of 3.
- `R/PipeOpTorchConvTranspose.R:81` — `` ::`integer()` `` missing space after `::`.
- `R/PipeOpTorchAvgPool.R:89` — stray outer parens: `` :: (`integer()`) ``.
- `NEWS.md:53-54` — continuation line at column 0 splits the bullet into two markdown blocks.
- `NEWS.md:55` — missing final period.

### 3.7 bibentries
- `R/bibentries.R:12-13` — stray trailing comma in the `bibentry()` call.
- `arik2021tabnet` (`:41`) and `ioffe2015batch` (`:53`) are referenced by no `format_bib()` call — dead weight.

### 3.8 "allows to X" (non-native phrasing) — 12 instances
`R/cache.R:4`, `R/PipeOpModule.R:7`, `R/PipeOpTorchBlock.R:14`, `R/DataDescriptor.R:6`,
`R/LearnerTorch.R:25`, `man-roxygen/paramset_torchlearner.R:101`, `README.Rmd:54`, `README.Rmd:114`,
`NEWS.md:108`, `NEWS.md:109`, `NEWS.md:171`, `DESCRIPTION:35`.
Left alone deliberately — `DESCRIPTION` is CRAN-visible and this is a style call. Suggested:
"allows one to X" / "makes it possible to X".

---

## 4. Documentation — missing entirely

Ordered by user impact. These are content tasks.

### 4.1 `jit_trace` is documented nowhere
Defined at `R/paramset_torchlearner.R:125`, present on every non-`LearnerTorchModule` learner and on
`po("torch_model_*")`, headlined in NEWS 0.2.0 ("can lead to significant speedups") — yet
`grep -rn "jit_trace" man/ man-roxygen/` returns **zero hits**, and it appears in no vignette or README.
Should live in `man-roxygen/paramset_torchlearner.R` plus the *Important Runtime Considerations*
section of `R/LearnerTorch.R`. Needs: that it runs `torch::jit_trace()` once at the start of training;
that it removes per-batch R interpreter overhead; that it requires data-independent control flow
(no data-dependent branching, no varying shapes); its batch-norm history (fixed in 0.2.1); and the
relation to `jittable`.

### 4.2 Marshaling is reference-only, and is a silent-data-loss footgun
`?LearnerTorch` has a four-line *Saving a Learner* section; that is the only place in the doc set.
`marshal`, `unmarshal`, `saveRDS`, `future`, `mirai` appear in **zero** README or vignette lines,
though `future`/`mirai`/`callr` are all in `Suggests`.

Confirmed empirically: `saveRDS()` of a trained but unmarshaled learner **succeeds**, then predicting
after `readRDS()` fails with `external pointer is not valid`.

Should cover: why torch external pointers don't survive serialization; the `$marshal()`/`$unmarshal()`
round-trip with a worked `saveRDS`/`readRDS` example; that `resample()`/`benchmark()` with
`future`/`mirai` marshal automatically via mlr3's `marshal` property, so the manual call is only for
hand-rolled `saveRDS`; and that `$model$network` is a live torch object that must not be shared
across processes.

### 4.3 No prose on validation, early stopping or internal tuning
`patience`, `min_delta`, `internal_valid_scores`, `$internal_tuned_values`,
`to_tune(..., internal = TRUE)`, `AutoTuner`, `set_validate()` with `"test"`/`"predefined"` —
**zero hits** across `README.Rmd` and all vignettes, despite the README bullet promising that networks
"can be easily tuned via `mlr3tuning` and friends". There is no worked `resample()`, `benchmark()` or
`AutoTuner` example anywhere.

`CallbackSetEarlyStopping` (`R/CallbackSetEarlyStopping.R`) is **not exported, not in
`mlr3torch_callbacks`, and has no man page** — early stopping is only reachable through the
`patience`/`min_delta` hyperparameters, which is stated nowhere a user would look.

Proposed: a new `vignettes/articles/tuning.Rmd` — set validation three ways → read
`internal_valid_scores` → configure `patience`/`min_delta` and show `$model$epochs` shrinking →
`epochs = to_tune(upper = 100, internal = TRUE)` combined with an external search space in an
`AutoTuner`, and why that is cheaper than tuning `epochs` externally.

### 4.4 `augment_` vs `trafo_` convention is explained once, in the wrong place
The convention carries real behaviour: `?mlr_pipeops_preproc_torch` states that `stages` "is set to
`"train"` when the `PipeOp`'s id starts with `"augment_"` and to `"both"` otherwise". That sentence is
the only statement of it, and it lives on the base-class page. In the vignettes, `stages` gets one
paragraph inside `lazy_tensor.Rmd`'s *Custom Preprocessing* section — an authoring tutorial, not a
usage one — and never lists all three levels.

Also: the 23 `trafo_*`/`augment_*` man pages have **no `\description{}` at all** (they were also
missing from the pkgdown index, which is fixed on this branch). At minimum each needs a one-line
description plus `@seealso` to the corresponding `torchvision::transform_*`.

### 4.5 Runtime/performance knobs are a parameter list, not guidance
`num_threads`, `num_interop_threads`, `tensor_dataset`, `num_workers`, `pin_memory`, `worker_globals`,
`worker_packages`, `worker_init_fn`, `collate_fn`, `timeout`, `sampler`, `batch_sampler`,
`batch_size_predict`, `shuffle`, `device`/`auto_device` — all defined in
`man-roxygen/paramset_torchlearner.R` and rendered only into `?LearnerTorch`'s parameter list.
**Zero** occurrences in README or any vignette.

`device = "cuda"` appears once (`pipeop_torch.Rmd:147`) and is silently overridden back to `"cpu"` in
a hidden chunk, so no reader ever sees a GPU discussion. `auto_device()` is exported with a
27-character description and never mentioned in prose. `?LearnerTorch`'s *Important Runtime
Considerations* is four bullets.

### 4.6 Authoring custom losses, optimizers and LR schedulers is undocumented
`get_started.Rmd` covers only *consuming* `t_loss()`/`t_opt()`. `TorchLoss$new()`, `as_torch_loss()`,
`TorchOptimizer$new()`, `as_torch_optimizer()`, `as_lr_scheduler()` appear in **no** vignette or README
line — `as_lr_scheduler` and `lr_scheduler` have zero hits across the narrative docs, even though six
of the eleven built-in callbacks are LR schedulers. Working examples exist only in `@examples`.

Also undocumented in prose: the `loss.` / `opt.` / `cb.<id>.` parameter-prefix convention (currently
one clause at `man/mlr_learners_torch.Rd:116`), and `task_types` filtering.

Proposed: extend `vignettes/articles/callbacks.Rmd` into "Customizing Training", also covering
`callback_set()` vs `torch_callback()` and terminating training via `ctx$terminate` (the
*Terminate Training* section in `R/CallbackSet.R` has no vignette counterpart).

### 4.7 Writing a custom `PipeOpTorch` has a reference contract but no tutorial
`?mlr_pipeops_torch` has a solid *Inheriting* section, but no vignette ever shows
`R6Class(inherit = PipeOpTorch)`; `internals_pipeop_torch.Rmd` covers `ModelDescriptor` from the
consumer side only.

Meanwhile the dev version exports 13 new shape helpers (`infer_shapes`, `broadcast_shapes`,
`resolve_dim`, `shape_to_str`, `assert_shape`, `assert_shapes`, `assert_known_dims`, `assert_ndim`,
`assert_same_ndim`, `assert_same_batch_size`, `assert_dim_in_range`, `assert_not_batch_dim`,
`assert_positive_extent`, `reshape_output_shape`) as bare reference stubs, and **removes the
`only_batch_unknown` constructor argument** — a breaking change for exactly the audience that has no
tutorial, with no migration note.

Proposed: `vignettes/articles/custom_pipeop_torch.Rmd` under the existing "Internals" menu group —
implement one non-trivial layer end to end, show `.shapes_out()` handling `NA` in *any* dimension via
`assert_known_dims()`, show registration so `nn("mykey")` works, and add the `only_batch_unknown`
migration note.

### 4.8 Vignettes don't ship with the package
`.Rbuildignore` contains `^vignettes/articles$` and `DESCRIPTION` has **no** `VignetteBuilder` field,
so the installed/CRAN package has no vignettes: `browseVignettes("mlr3torch")` is empty and
`?mlr3torch` offers no entry point.

Compounding this, **no man page in `man/` contains a `vignette(` call or an `articles/` link** — the
reference docs and the website tutorials are two disconnected islands. The README's only pointer is
the unlinked sentence "Start by reading one of the vignettes on the package website!"
(`README.Rmd:211`).

Fix: either add `VignetteBuilder: knitr` and un-ignore `vignettes/articles`, or accept website-only
but add real article URLs to the README and `@seealso` links from `?LearnerTorch`, `?lazy_tensor`,
`?mlr_pipeops_torch`, `?TorchCallback`.

### 4.9 Caching and downloads
- `options(mlr3torch.cifar_download_timeout = <seconds>)` is read at `R/TaskClassif_cifar.R:60`
  (default 5400s) and exists **only as a source comment** — no man page, no `@section Options`.
- `mlr3torch.cache` is documented in `?mlr3torch-package` but has **zero** hits in README and vignettes,
  even though `lazy_tensor.Rmd` and `pipeop_torch.Rmd` silently trigger MNIST/tiny-imagenet downloads
  (`pipeop_torch.Rmd` hides the download in an `include = FALSE` chunk).
- No task man page states its download size.
- The cache-version invalidation mechanism (`CACHE$versions`, `version.json`, and the error
  "Cache directory '%s' was not initialized by mlr3torch" at `R/cache.R:60`) is `@noRd`, so a user who
  hits that error has **no documented recovery**.

### 4.10 No fine-tuning / transfer-learning documentation
Despite 53 pretrained vision learners. `vision_learner_list.Rmd` is a bare table; `pretrained` gets
one sentence on `?mlr_learners.torchvision`. `replace_head()`'s description is a duplicated sentence
fragment ("Replace the head of a network Replace the head of the network with a linear layer with
d_out classes."). `CallbackSetUnfreeze` / `t_clbk("unfreeze")` — the intended companion for staged
fine-tuning — appears in no vignette. Nothing documents where pretrained weights are downloaded to,
how large they are, or the input-size/normalization requirements per architecture.

### 4.11 `seed` semantics are misleadingly documented
Both README and `get_started.Rmd` demonstrate reproducibility with plain `set.seed()`. The actual
`seed` hyperparameter — `integer(1)` | `"random"` | `NULL`, initialised to `"random"`, stored in
`$model$seed`, **with the consequence that clones of a learner use a different seed** — is documented
only in `man-roxygen/paramset_torchlearner.R:21-28` and appears in no narrative doc.

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
`lr_step`) have no `mlr_callback_set.<key>` topic and land on a generic page with no Parameters section.

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
  `lazy_tensor` task — the motivating use case `lazy_tensor.Rmd` asserts but never demonstrates —
  plus a multi-modal `tsk("melanoma")` example (tabular + `lazy_tensor` in one task, a
  README-advertised feature with no example anywhere).
- **`custom_pipeop_torch.Rmd` — "Writing your own PipeOpTorch"** (4.7).
- Extend **`callbacks.Rmd` → "Customizing Training"** (4.6).

---

## 5. UX / polish

### 5.1 No hint to marshal on `external pointer is not valid`
`saveRDS()` of an unmarshaled trained learner succeeds; predicting after `readRDS()` then fails with
`external pointer is not valid`. Technically correct, and the documented flow is marshal-first, but a
targeted message ("did you forget `$marshal()`?") would save real time. See also 4.2.

### 5.2 `t_clbk("early_stopping")` gives an unhelpful error
`Element with key 'early_stopping' not found in DictionaryMlr3torchCallbacks!` — without listing valid
keys or mentioning that early stopping is configured via the learner's `patience`/`min_delta`
parameters rather than a dictionary callback. A plausible thing for a user to get wrong.

### 5.3 Predicting on a marshaled learner silently works
`$marshaled` stays `TRUE` and `model$network` is `NULL`, yet prediction returns bit-identical results
(mlr3 unmarshals internally). Harmless, but reads as inconsistent.

### 5.4 [D] A hand-built graph and `lrn("classif.mlp")` don't match at equal seed
Same architecture, same seed, same optimizer/loss/batch_size/epochs. Both networks are structurally
identical (83 params, shapes `10x4 | 10 | 3x10 | 3`), yet **initial weights already differ at
`epochs = 0`**, so predictions diverge (max abs diff 0.18 with `shuffle = FALSE`).

Reproducibility *within* each construction path is exact (same seed → bit-identical, different seed →
different). What fails is equality *across* construction paths — presumably the torch RNG is seeded at
a different point relative to module construction in `LearnerTorchMLP` vs the
`PipeOpTorch`/`ModelDescriptor` path.

Worth a decision because "does my hand-built graph reproduce the built-in learner?" is the natural way
a user validates a custom architecture, and it silently answers no.

Minor related detail: `classif.mlp` always inserts an `nn_dropout` module (param names `0`, `3` ⇒
slots 0..3) even at `p = 0`.

---

## Appendix — verified clean, don't re-audit

**Docs**: all 124 runnable examples pass (extracted via `tools::Rd2ex` with `\dontrun`/`\donttest`
uncommented, each run in a fresh session); all 6387 cross-references resolve; `tools::checkDocFiles()`
clean; 142 R6 method blocks have `@param` == formals in both directions; no `@field` documents a
non-existent field; ParamSet ids match prose everywhere except the cases listed above; all `@inherit`
targets exist; `man/` is byte-identical to a fresh `roxygenise()`; all dev-version NEWS entries verified
true at runtime except 3.2.

**Code**: marshal/unmarshal round-trips bit-identical (including `jit_trace = TRUE` / `script_module`);
row ordering safe (`.dataloader_predict()` forces `shuffle = FALSE`, `drop_last = FALSE` and drops
`sampler`/`batch_sampler`; batched predictions on a filtered, reordered task match per-row predictions;
`tensor_dataset = TRUE` matches `FALSE`); `network$eval()` correctly set at predict time; `pointer_shape`
vs `pointer_shape_predict` guard correctly rejects train-only augmentation; seed sampled once and reused
at predict; `clone(deep = TRUE)` fully independent; `lazy_tensor` columns sharing a dataset materialize
independently; shape inference correct under all 2ⁿ unknown-dim patterns (see 2.1).

**Empirical**: 135 configuration checks across 8 areas, 131 passed. All 8 learner types on
binary/multiclass/regression (probs sum to 1, column order matches `class_names`,
`response == argmax(prob)`, factor levels preserved incl. classes absent from training); every
documented parameter demonstrably takes effect; graph building incl. merges, blocks, `nn_fn`, reshape
round-trips; `materialize`/`DataDescriptor`; `trafo_normalize` exact and train/predict-identical;
`augment_hflip` correctly train-only; all callbacks; all four `validate` modes;
`resample`/`benchmark`/`AutoTuner`/`tnr("internal")`; `po("learner")` in graphs; `GraphLearner` marshal
round-trip; callr encapsulation; all 17 edge cases (single feature, single-row predict, constant
feature, `batch_size` > n, `predict_newdata`, clear errors for factor-only features and missing values).
