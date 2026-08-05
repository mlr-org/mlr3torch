# mlr3torch — open TODOs from the audit

Everything here is **still open**. Items already fixed on branch `audit-fixes` are listed in `NEWS.md`
and are not repeated below.

Source: an 8-agent audit of `main` @ `736f5165` — adversarial bug hunt, doc/code correctness,
empirical usage sweep (135 configuration checks), doc coverage.

**[D]** marks items that need a decision from you, not just a patch.

---

## 1. Bugs — code

### 1.0 [D] **`rr$learners[[i]]` is silently the wrong learner** — highest-impact finding
`R/LearnerTorch.R:739`:
```r
hash_input.nn_module = function(x) {
  data.table::address(x)
}
```

A memory address is not stable across copies, so two `identical()` learners hash differently. This
breaks the contract mlr3 relies on, and the consequence is **silently wrong results from a documented
public API**.

```r
task = tsk("iris"); set.seed(1)
rr = resample(task, lrn("classif.mlp", epochs = 2, batch_size = 32, neurons = 10,
                        device = "cpu", predict_type = "prob"),
              rsmp("cv", folds = 3), store_models = TRUE)
# max abs difference between learner i's predictions and fold j's stored predictions
#         [,1]   [,2]   [,3]
# [1,]  0.3840 0.3987 0.0000
# [2,]  0.7614 0.0000 0.3911
# [3,]  0.0000 0.7896 0.4000
```
The zeros should lie on the diagonal; instead the list is reversed. `lrn("classif.rpart")` under the
identical harness gives a clean zero diagonal, so this is specific to mlr3torch.

**Mechanism.** `classif.mlp`'s `activation` parameter holds an `nn_relu` generator. Each resampling
iteration copies it, so the copies are `identical()` but sit at different addresses:
```
addresses: 0x555560b1f528  0x5555611a70f0  0x5555614f8040
identical(learners[[1]]$param_set$values$activation, learners[[2]]$...):  TRUE
```
Three different `learner_hash` values then feed `ResultData$learners()`, which does
`merge(tab, learner_components, by = "learner_hash", sort = TRUE)` — so `$learners` comes back
ordered by an address-derived hash rather than by iteration.

**It is nondeterministic**, since it depends on allocation order: some parameter settings happen to
produce the right order. That is exactly the profile of a bug that passes CI most days.

**Scope:** `rr$learners` and `bmr$learners` are affected. `rr$score()` and `as.data.table(rr)` were
correct in every run checked — they do not go through the sorted merge. So the damage is confined to
anyone inspecting a fold's model, e.g. `rr$learners[[1]]$model`.

**Why this needs your decision rather than a quick patch:** the obvious replacement — letting
`mlr3misc:::hash_input.function` handle it, i.e. `list(formals(x), as.character(body(x)))` — is what
this method was written to override, and it is wrong here: every generator produced by
`torch::nn_module()` shares the same wrapper body and formals, so `nn_relu` and `nn_sigmoid` would
hash **equal**. Trading a copy-instability bug for a collision bug is not an improvement.

A workable direction is to key on the generator's class, which is stable across copies and distinct
per module type (`class(nn_relu)[1]` is `"nn_relu"`, `class(nn_sigmoid)[1]` is `"nn_sigmoid"`), with
a fallback for anonymous modules built by `nn_module(classname = NULL)`, whose class is just
`c("nn_module", "nn_module_generator")` — those would need to fall back to hashing the
`initialize`/`forward` methods to stay distinguishable. Deciding how far to go is a judgement call
about what identity should mean for an `nn_module` hyperparameter, which is why it is left here.

Worth adding a regression test that `resample(..., store_models = TRUE)$learners[[i]]` reproduces
iteration `i`'s predictions, for a learner whose parameters hold an `nn_module`.

### 1.1 [D] Checkpoint callback rejects a directory that already holds checkpoints
`R/CallbackSetCheckpoint.R:51` — `if (is_empty_dir(path)) path else assert_path_for_output(path)`.

Partially addressed on `main` (75a19f815): an existing *empty* directory is now accepted, which covers
a pre-created output folder and a run that died before its first checkpoint. The resampling case is
still open — once iteration 1 has written `network1.pt` the directory is no longer empty, so
checkpointing remains **unusable with `resample()`, `benchmark()` and tuning**, and training the same
learner twice into the same path still fails.

```r
d = tempfile()
lx = lrn("classif.mlp", epochs = 1, batch_size = 32, neurons = 5,
  callbacks = t_clbk("checkpoint", path = d, freq = 1))
resample(tsk("iris"), lx, rsmp("cv", folds = 2))
#> Error: Assertion on 'path' failed: File at path already exists
```

*Why this needs a decision:* `tests/testthat/test_CallbackSetCheckpoint.R:38` explicitly asserts the
error for a directory holding unrelated data, so guarding against overwriting foreign data is
deliberate; the question is only how a *second run of the same learner* should behave. Options:
- (a) `overwrite = TRUE` and relax the test;
- (b) keep the guard but write into a per-iteration subdirectory so resampling works;
- (c) accept a directory that contains only `network*.pt` / `optimizer*.pt` (i.e. a previous run's
  checkpoints) and overwrite those;
- (d) keep as-is and document the limitation prominently.

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

**Root cause is a `torch` bug, not an mlr3torch decision.** `torch`'s `R/with-indices.R` exists to
translate 1-based indexing into libtorch's 0-based one. `torch_cross_entropy_loss()` converts
`target` via `to_index_tensor()` but forwards `ignore_index` unchanged, even though libtorch compares
it against the already-converted target. The same omission is in `torch_nll_loss()`,
`torch_nll_loss2d()` and `torch_nll_loss_nd()`. Nothing caught it because the default `-100` is a
sentinel that matches no target either way, and `torch` has no test for the argument at all.

**Decided: do not work around it in mlr3torch** — a shift here would have to be removed again once
`torch` fixes it. Documented in a comment at the parameter in `R/TorchLoss.R`. Report upstream.

Still open here regardless: nothing validates `ignore_index` against `seq_along(task$class_names)`,
so an out-of-range value silently ignores nothing.

### 1.3 [D] `internal_valid_scores` reports the last epoch, `internal_tuned_values` the best one
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

Two independent audit rounds found this. The second round established that **it deviates from the
ecosystem convention and changes tuning results**, so it is a genuine bug rather than a convention:

- `mlr3learners`' xgboost indexes its evaluation log at `attributes(model)$early_stop$best_iteration`,
  i.e. exactly the iteration reported in `internal_tuned_values`. Verified directly by printing
  `mlr3learners:::.__LearnerClassifXgboost__.extract_internal_valid_scores`. (An earlier audit pass
  claimed xgboost behaves like mlr3torch — that claim was wrong.)
- `tnr("internal")` + `msr("internal_valid_score")` ranks external configurations by this number, so
  it ranks them by the score of models that are then discarded. A **reproducible rank flip** was
  demonstrated on sonar (seed 4, `patience = 2`): reported scores pick `lr = 0.005`, but at each
  configuration's own tuned epoch `lr = 0.02` is better (0.2097 vs 0.2258).
- With `classif.ft_transformer` the gap reached an order of magnitude: best epoch 3 scored 0.0222
  while the reported `internal_valid_score` was 0.3333 from epoch 5.
- It does not even require early stopping to fire — with `patience = 100` and 4 epochs, the tuned
  value is epoch 2 (score 0.289) while the reported score is epoch 4's (0.378).

The defensible reading is that `patience`'s docs say "the final model is stored in the learner, not
the best model", so the score does describe the stored network — but then the two extractors describe
two different models while mlr3tuning pairs them as if they described one.

**Decision needed** on which way to resolve it: (a) restore the best-epoch weights, (b) record the
best-epoch score, or (c) at minimum document loudly that `internal_valid_score` must not be used as
the tuning measure when `patience > 0`.

### 1.4 `CallbackSetHistory$load_state_dict()` is never called
`R/CallbackSetHistory.R:55-62`. Nothing in `R/` calls it, so the resume path is dead code — and
`state_dict()` returns `rbind(state, self$prev_state)`, i.e. new epochs *before* older ones, so the
ordering bug is baked in should it ever be wired up. Either finish it or delete it.
*(Static reasoning only.)*

### 1.5 `replace_head.mobilenet_v2` / `.VGG` hardcode the head input size
`R/LearnerTorchVision.R:168` (`nn_linear(1280, d_out)`) and `:188` (`nn_linear(4096, d_out)`), where
every sibling method reads `$in_features`. Correct only for the standard width variants.
*(Static reasoning only — verifying needs a model download.)*

### 1.6 [D] `seed` does not seed weight initialization for graph-built learners
A graph-built learner is **not reproducible**, even with an explicit `seed`, while the equivalent
predefined learner is. `resample()` on such a learner gives different results run to run.

```r
mk = function() as_learner(
  po("torch_ingress_num") %>>% po("nn_head") %>>%
  po("torch_loss", t_loss("cross_entropy")) %>>% po("torch_optimizer", t_opt("adam")) %>>%
  po("torch_model_classif", batch_size = 50, epochs = 1, seed = 1L, predict_type = "prob"))

{ l = mk(); l$train(tsk("iris")); l$predict(tsk("iris"))$prob[1, ] }  # 0.979 0.004 0.017
{ l = mk(); l$train(tsk("iris")); l$predict(tsk("iris"))$prob[1, ] }  # 0.066 0.010 0.925  <- differs
```

**Mechanism:** `PipeOpTorch$.train()` calls `.make_module()`, so the `nn_module`s are instantiated
while the *Graph* trains. `LearnerTorchModel$.network()` then returns the already-built network. All
weight initialization therefore happens **before** the `with_torch_settings(seed = ...)` block in
`LearnerTorch$.train()` (`R/LearnerTorch.R:521`). Predefined learners build their network inside
`.network()`, which runs within that block, so they are unaffected.

User-visible consequence: `?mlr_learners_torch` documents `seed` as "the torch seed that is used
during training and prediction", which is not true for this construction path. The only workaround
today is `torch::torch_manual_seed()` immediately before building the graph, which is undiscoverable
and does not survive `resample()`/`benchmark()`.

**Decision needed** because the fix is structural: either seed before the graph's `$train()`, or defer
module construction into the learner, or thread the seed through `ModelDescriptor`.

### 1.7 [D] `lrn("classif.ft_transformer")` cannot be trained with its documented defaults
```r
lrn("classif.ft_transformer", epochs = 1, batch_size = 150, device = "cpu")$train(tsk("iris"))
#> Error: Assertion on 'xs' failed: d_token: Must be of type 'single integerish value', not 'NULL'.
```
After supplying `d_token` and `n_blocks` you hit a second wall: exactly one of `ffn_d_hidden` /
`ffn_d_hidden_multiplier` must be set, and neither has an `init`.

`R/LearnerFTTransformer.R:55-56` declares `default = 3` for `n_blocks` and `default = 192` for
`d_token`, and `as.data.table(learner$param_set)` shows those defaults — but paradox `default`s are
documentation-only and nothing `init`s them. So three parameters are de-facto mandatory while the docs
present two of them as defaulted and never mention `ffn_d_hidden_multiplier` at all (it is inherited
from `PipeOpTorchFTTransformerBlock`). The man-page example sets all three, which hides the problem.

The errors also name "PipeOp block_1", an object the user never constructed, and don't name the learner.

`classif.tabm` does **not** have this problem. **Decision needed:** `init` these to the paper values,
or mark them required and error with a message naming the learner and the missing parameters.

### 1.8 [D] `d_token` must be divisible by `attention_n_heads`, but nothing declares or checks it
The most natural FT-Transformer search space is a landmine:

```r
l = lrn("classif.ft_transformer", epochs = 3, batch_size = 32, n_blocks = 1,
  ffn_d_hidden_multiplier = 2, device = "cpu")
l$param_set$set_values(d_token = to_tune(p_int(4, 16)), attention_n_heads = to_tune(1, 4))
tune(tnr("random_search"), tsk("iris"), l, rsmp("holdout"), msr("classif.ce"), term_evals = 3)
#> Error in value_error("embed_dim must be divisible by num_heads")
#> This happened in PipeOp block_1's $train()
```

`learner$param_set$deps` is empty and nothing validates the pair, so a raw torch `value_error` aborts
the entire `tune()`/`benchmark()` run. `encapsulate("evaluate", fallback = ...)` rescues it, but a new
user won't know to reach for that. The constraint appears nowhere in `?mlr_learners.ft_transformer`.

paradox cannot express "divisible by", so this needs a `custom_check` or a `.trafo`-level assertion
naming both parameters — hence a judgement call on where to put it — plus a line in the docs.

### 1.9 `attention_initialization` is a validated hyperparameter that does nothing
`R/PipeOpTorchFTTransformerBlock.R:170` declares
`attention_initialization = p_fct(levels = c("kaiming", "xavier"), init = "kaiming")`, documented as
"Initialization method for attention weights" and exposed on `lrn("classif.ft_transformer")`.
`nn_ft_transformer_block$initialize()` accepts it as a formal (`:56`) and **never references it in the
body** — `nn_multihead_attention()` is built with torch's own default init.

Confirmed: `grep -rn attention_initialization R/` finds only the declaration and the formal; the two
settings produce bit-identical `in_proj_weight` values under a fixed seed.

Every *other* block hyperparameter was swept and does reach the module (`ffn_d_hidden`,
`ffn_d_hidden_multiplier`, `ffn_dropout`, `residual_dropout`, `attention_dropout`, `attention_bias`,
`ffn_bias_first`, `ffn_bias_second`, `ffn_activation`, `prenormalization`, `is_first_layer`,
`query_idx`).

Not fixed here because implementing it means deciding what "kaiming"/"xavier" mean for the packed
`in_proj_weight` — a modelling decision that should match the FT-Transformer reference implementation.

### 1.9b Two exposed hyperparameter levels that always failed — **fixed upstream in `torch`**
`attention_bias = FALSE` on the FT-Transformer, and `anneal_strategy = "linear"` on
`t_clbk("lr_one_cycle")`, both failed unconditionally. Both root causes were in `torch` itself and are
fixed on the branch `fix/mha-bias-and-lr-one-cycle` of the `torch` checkout at `/Users/sebi/mlr/torch`:

- `nnf_multi_head_attention_forward()` left `k` and `v` unassigned when `bias = FALSE` and
  `query` differed from `key` — mlr3torch always hits this, since the last FT block sets
  `query_idx = -1`. Failed with `object 'k' not found`.
- `lr_one_cycle()` assigned its annealing function to `self.anneal_func` instead of
  `self$anneal_func`, so the linear strategy never set it. Failed with `attempt to apply non-function`.

**Nothing to do in mlr3torch** beyond depending on a `torch` version containing the fixes once they
are released. Until then both remain reachable levels that a tuner samples.

### 1.9d [D] FT-Transformer `n_blocks = 0` is permitted by the ParamSet but crashes
`p_int(lower = 0L)` explicitly allows it, so `n_blocks = to_tune(0, 4)` looks legal, but
`map(seq_len(0), ...)` yields an empty list which is then fed to `%>>%`:
```
Error: Assertion on 'class2' failed: Must have length 1.
```
Should either work (tokenizer → CLS → head) or be rejected with a message.

### 1.9e [D] FT-Transformer exposes `query_idx` and `is_first_layer`, which are always overwritten
Both are in the learner's `$param_set` and therefore tunable, but `.network()` sets them per block
(`query_idx = -1` for the last block, `NULL` otherwise; `is_first_layer` by position). Setting either
on the learner produces byte-identical trained weights. `query_idx`'s own help already says *"Should
not be set manually"* — in which case it arguably should not be reachable from the learner.

### 1.9f [D] Optimizer bounds are narrower than torch's actual domain
`weight_decay` is `p_dbl(upper = 1)` on all five optimizers and `eps` is `p_dbl(upper = 1e-4)` on
adam/adamw/adagrad, but torch accepts larger values:
```r
t_opt("adam", weight_decay = 2)   #> Error: weight_decay: Element 1 is not <= 1.
optim_ignite_adam(nn_linear(2, 1)$parameters, lr = 0.1, weight_decay = 2)   #> works
```
These look like accidental upper bounds and they block legitimate tuning ranges. Widening them is
harmless for existing code, but bounds are a maintainer's call.

### 1.9g [D] `t_clbk("history")` with no measures silently produces an empty table
```r
l = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 5, callbacks = t_clbk("history"))
l$train(tsk("iris")); l$model$callbacks$history
#> Empty data.table (0 rows and 1 cols): epoch
```
"Saves the training and validation history during training" reads like it will record *something*.
It only ever records `measures_train`/`measures_valid`; the training loss — the one quantity that
always exists — is never logged. A warning when `history` is enabled with no measures, or logging the
loss by default, would save a confused debugging session.

### 1.9h [D] `num_interop_threads` is unusable, and permanently alters global torch state
The parameter is *initialised* to `1`, and torch only allows the interop thread count to be set once
per session — so training any learner at all burns that one chance, after which the user's explicit
value is rejected:

```r
torch::torch_get_num_interop_threads()   #> 8
lrn("classif.mlp", epochs = 1, batch_size = 32, neurons = 8)$train(tsk("iris"))
torch::torch_get_num_interop_threads()   #> 1   (and never restored)
lrn("classif.mlp", ..., num_interop_threads = 4)$train(tsk("iris"))
#> WARN Can only set the interop threads once, keeping the previous value 1
```

It only works if the very first learner trained in the session already carries the desired value.
Two further problems:
- Unlike `num_threads`, which `R/with_torch_settings.R` restores via `on.exit`, the interop setting is
  never restored — so mlr3torch permanently drops interop parallelism to 1 for **all unrelated torch
  code** in the session.
- The docs say only that it "can only be set once during a session"; they do not say the package's own
  default has already consumed that chance.

Also a plain doc error at `man-roxygen/paramset_torchlearner.R:17` (rendered into
`man/mlr_learners_torch.Rd` and `man/mlr_pipeops_torch_model.Rd`): `num_interop_threads` is described
as "The number of threads for intraop **and** interop pararallelization" — it is interop only, and
"pararallelization" is misspelled. *(The typo is fixed on this branch; the semantics and the
initialise-to-1 decision are left to you.)*

### 1.9i [D] `mirai` backend silently produces irreproducible results
```r
daemons(2, .compute = "mlr3_parallelization")
set.seed(42); b = resample(tsk("sonar"), mk(), rsmp("cv", folds = 3))
set.seed(42); c = resample(tsk("sonar"), mk(), rsmp("cv", folds = 3))
identical(gp(b), gp(c))   #> FALSE      (TRUE under future)
```
Root cause is upstream: `mlr3:::future_map`'s mirai branch calls `mirai_map()` without seeding, while
the `future` branch passes `future.seed = TRUE`. `classif.ranger` reproduces it, so it is not
mlr3torch-specific. It is recorded here because **mlr3torch learners are stochastic by default**
(`seed = "random"`), so torch users are maximally exposed, and nothing in the docs suggests pinning
`seed` before going parallel. Setting the learner's `seed` explicitly makes mirai reproducible and
equal to the sequential result. Worth reporting upstream and mentioning in the parallelization docs.

### 1.9j `device = "cuda"` on a CPU-only build dumps a 60-frame C++ backtrace
`auto_device()` resolves `"auto"` but passes an explicit `"cuda"` through unchecked, so the failure
arrives as ~60 frames of C++ backtrace that also land verbatim in `learner$log` and `rr$errors`,
making resampling logs unreadable. A `torch::cuda_is_available()` check at train time would reduce
this to one actionable line.

### 1.10 No check that the final layer size matches the task
Forgetting `po("nn_head")`, or giving the last layer the wrong `out_features`, is not caught even
though the number of classes is known at `$train()` time.

- Too few outputs fails mid-training with a raw ATen error:
  `Target 3 is out of bounds. Exception raised from nll_loss_out_frame`.
- Too many outputs **trains successfully** and fails at predict with
  `Error in dimnames(x) <- dn : length of 'dimnames' [2] not equal to array extent`, or with
  `Assertion on 'response' failed: Contains missing values (element 1)` for `predict_type = "response"`.

None of the three messages would lead a user to "your network's output dimension doesn't match the
task". A check in `PipeOpTorchModel$.train()` comparing `pointer_shape` against the task's output
dimension would catch all three cases up front.

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

### 2.2 No checkpoint test writes into a path that a previous run already checkpointed into
Existing paths are covered (empty, and holding unrelated data), but never a second run of the same
learner or a `resample()`, which is why 1.1 is invisible to CI.

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

### 3.3 [D] Two unused bibentries
`arik2021tabnet` and `ioffe2015batch` in `R/bibentries.R` are referenced by no `format_bib()` call.
Dead weight, but deleting a reference is a judgement call — they may be intended for learners not yet
written up.

### 3.4 "allows to X" (non-native phrasing) — 12 instances
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

### 4.3b Multi-modal graphs need a `po("select")` per ingress, shown nowhere
`gunion(list(po("torch_ingress_num"), po("torch_ingress_categ"), po("torch_ingress_ltnsr")))` on a
task with all three feature types does not work — each ingress rejects a task containing any other
feature type. The error is actionable (`Consider using po("select")`) and the
`po("select", selector = selector_type(...))` version trains fine, but no vignette shows the
multi-modal pattern, and it is the natural thing to reach for after reading about the three ingress
flavours. This is the same README-advertised feature noted in 4.15.

Related gap: `$shapes_out()` exists only on individual `PipeOp`s, not on a `Graph`
(`g$shapes_out(...)` gives `attempt to apply non-function`). The docs correctly scope it to a single
`PipeOp`, so this is a missing convenience rather than a contradiction — but it is the obvious thing
to want when debugging a long chain.

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

### 5.0 [D] Out-of-the-box `lrn("classif.mlp")` loses to `featureless`
```r
benchmark(benchmark_grid(tsk("iris"),
  list(lrn("classif.mlp", epochs = 20, batch_size = 32, neurons = 10, device = "cpu"),
       lrn("classif.featureless")), rsmp("cv", folds = 2)))$aggregate(msr("classif.ce"))
#> classif.mlp 0.793 | classif.featureless 0.780
```
Cause: `p` (dropout) is initialised to `0.5`. With `p = 0` the same learner scores 0.04 against
featureless' 0.673. The default is documented, but 50% dropout is an unusual out-of-the-box value and
it makes the first thing a new user runs look broken. Similarly `regr.mlp` on unscaled `mtcars` loses
badly to `regr.featureless` (MSE 168 vs 35) — expected without `po("scale")`, but unlike the
FT-Transformer docs the MLP docs never mention scaling. Changing an initialised default is your call.

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

## Appendix A — investigated and rejected

Recorded so they are not re-reported as findings.

- **`t_loss("cross_entropy", ignore_index = NULL)` erroring is correct.** `ignore_index` is a `p_int`,
  so paradox rejects `NULL` with a message that names the parameter and the expected type. This is
  unlike `class_weight`, a `p_uty` whose documented default *is* `NULL` — that case was a real bug and
  is fixed on this branch.

- **`nn_squeeze` squeezing the batch dimension is intended, not a bug.** An audit agent flagged that
  `po("nn_squeeze", dim = 1)$shapes_out(list(c(1, 4, 3)))` returns `c(4, 3)` while every sibling
  operator that changes a dimension guards with `assert_not_batch_dim()`. Adding that guard breaks
  `tests/testthat/test_PipeOpTorchReshape.R:146`, which asserts the current behaviour *and* verifies
  on line 149 that the module itself does the same thing at runtime
  (`dim(nn_squeeze(dim = 1L)(torch_randn(1, 3, 5)))` is `c(3, 5)`). The shape inference therefore
  correctly mirrors torch, and a batch dimension that is not known to be `1` is kept anyway (line 151).
  The `assert_not_batch_dim()` example in `R/shape.R:450` happens to use `id = "nn_squeeze"`, which is
  misleading — it is illustrative only.

- **The default optimizer is Adam, and the code docs saying so are correct.** See 3.2 — only the NEWS
  entry is wrong.

---

## Appendix B — verified clean, don't re-audit

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
