# Changelog

## mlr3torch (development version)

### Breaking changes

- The `freq_type` parameter of `t_clbk("checkpoint")` was removed and
  checkpointing is always per epoch.
- The construction argument `only_batch_unknown` of `PipeOpTorch` was
  removed, as shape inference functions are now expected to handle
  multiple unknown dimensions.
- The dropout probability `p` of `lrn("classif.mlp")` /
  `lrn("regr.mlp")` is now initialized to `0.1` instead of `0.5`. Set
  `p = 0.5` explicitly to keep the old behaviour.
- The `cache` argument of
  [`materialize()`](https://mlr3torch.mlr-org.com/dev/reference/materialize.md)
  is now a [`utils::hashtab()`](https://rdrr.io/r/utils/hashtab.html)
  instead of an
  [`environment()`](https://rdrr.io/r/base/environment.html), which
  avoids possible hash collisions and raises the R dependency to
  `>= 4.2.0`.
- The first argument of the private `.encode_prediction()` method of
  `LearnerTorch` was renamed from `predict_tensor` to `network_output`,
  as it can also be a [`list()`](https://rdrr.io/r/base/list.html) of
  tensors.
- `measures_train` is now calculated from the complete output of the
  network instead of only its first tensor, which is passed to
  `.encode_prediction()` unchanged.
- The `num_interop_threads` parameter of `LearnerTorch` is no longer
  initialized to `1`, so torch’s default is left in place unless the
  parameter is set. Setting it to a value that torch can no longer apply
  is now an error instead of a warning.

### Features

- `LearnerTorch` and `PipeOpTorchModel` now accept any task type
  registered in `mlr_reflections$task_types`, via the new generics
  [`get_target_batchgetter()`](https://mlr3torch.mlr-org.com/dev/reference/get_target_batchgetter.md)
  and
  [`encode_prediction()`](https://mlr3torch.mlr-org.com/dev/reference/encode_prediction.md).
- New S3 generic
  [`get_batch_constructor()`](https://mlr3torch.mlr-org.com/dev/reference/get_batch_constructor.md),
  which decides how a whole batch of a task is built, i.e. both the
  features `x` and the target `y`.
- A network can now return a
  [`list()`](https://rdrr.io/r/base/list.html) of tensors in evaluation
  mode, which is passed to
  [`encode_prediction()`](https://mlr3torch.mlr-org.com/dev/reference/encode_prediction.md)
  as it is, so a prediction can consist of more than one quantity.
- New function
  [`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md)
  that simplifies the creation of `PipeOpTorch` classes.
- New article *Writing your own PipeOpTorch*.
- The `$model` of a `LearnerTorch` now has a printer.
- Added more image learners from {torchvision}.
- Most `LearnerTorchVision` are now `jittable`.
- Ported the `TabM` tabular learner from Python.
- `LearnerTorch` now has `.loss_fn(task, param_vals)` private method
  that allows to customize the construction of the loss function.
- `LearnerTorch` now has `restore_best_weights` parameter that can be
  used when early stopping is active.
- A network can now return a
  [`list()`](https://rdrr.io/r/base/list.html) of tensors during
  training, which the loss is applied to. In `ContextTorch`, `$y_hats`
  is that complete output and `$y_hat` its first element.
- New parameter `batch_size_predict` for `LearnerTorch`, which overrides
  `batch_size` for prediction
- Added multihead attention and transformer encoder pipeops.
- Any dimension of an input shape can now be unknown (`NA`), not only
  the batch dimension.
- Improved error messages during `PipeOpTorch`’s shape inference.
- The `shape` parameter of `nn("reshape")` can now be a
  `function(shape)` of the input shape.
- Exported various helpers useful for implementing shape inference for
  custom `PipeOpTorch` classes.
- `ContextTorch` has a new field `$callbacks`, which gives a callback
  access to the other callbacks of the training run.
- Callbacks can now be ordered via a `weight` field.
- A `LearnerTorch` can now be resumed from a checkpoint, which includes
  resuming the callbacks.
- `t_clbk("checkpoint")` now also writes the callback states, as well as
  the class behind each callback id, so a resumed run errors instead of
  restoring a state into a different callback.
- The `path` of `t_clbk("checkpoint")` can now be a `function()` that is
  called at the beginning of each training run and returns that run’s
  path.
- `t_clbk("checkpoint")` now checks each file immediately before writing
  it, so two runs writing into one folder error instead of mixing their
  checkpoints.
- Resuming a checkpoint that is already at `epochs` now returns its
  model instead of erroring, so a script that restarts itself can be run
  again after it succeeded.
- `t_clbk("checkpoint")` now reports its folder in
  `learner$model$callbacks$<id>$path`, so the folder a `path` function
  chose can be read off the trained learner.
- Resuming a run that early stopping had ended now warns and returns its
  model instead of training further, and `ctx$terminate` is checked
  before an epoch rather than after it.
- `t_clbk("progress")` now prints the epoch as `Epoch <n>/<epochs>`, so
  a resumed run shows how much of it is left.

### Bug fixes

- `lrn("classif.torch_model")` / `lrn("regr.torch_model")` no longer
  change their `$hash` when they are trained.
- Fixed some hashing bugs related to R jit compilation.
- `ContextTorch$epoch` is now `0` during the `on_begin` stage instead of
  `NULL`.
- [`replace_head()`](https://mlr3torch.mlr-org.com/dev/reference/replace_head.md)
  for `mobilenet_v2` and `VGG` works for `width_mult` above 1.
- `PipeOpTorch$shapes_out()` now always returns
  [`integer()`](https://rdrr.io/r/base/integer.html) shapes (and not
  sometimes doubles like `NA`).
- `po("torch_model_classif")` and `po("torch_model_regr")` now have the
  correct `$packages`.
- The `batch_sampler` parameter can now be used without setting
  `batch_size` for training.
- Configuration errors that are only caught during `LearnerTorch` no
  longer trigger a fallback learner.
- The `LearnerTorch`’s `sampler` and `batch_sampler` parameters are now
  not used during prediction.
- [`logical()`](https://rdrr.io/r/base/logical.html) features are now
  encoded as 1-based instead of 1-based.
- `lazy_tensor` columns are now again printed correctly inside
  `data.table`s
- Fixed some links on the pkgdown website and the help pages.
- Fixed various other shape inference bugs.
- `po("torch_model_{regr, classif}")` now resets the parameters of the
  network at the beginning of `$train()` when the network is built from
  `PipeOpTorch` objects, which makes the results reproducible for the
  set `seed` parameter.
- [`nn()`](https://mlr3torch.mlr-org.com/dev/reference/nn.md) now
  properly interprets `nn("linear_1")` as
  `po("nn_linear", id = "linear_21")`.
- Fixed some bugs in `FTTransformer`: `attention_initialization` now has
  an effect, `n_blocks = 0` is allowed and the hidden dimension falls
  back to `d_token * 4/3` as in the reference implementation.
- Fixed some issues in the documentation.
- Examples, vignettes and the README now use `nn("linear")` instead of
  the equivalent, but longer `po("nn_linear")`.

## mlr3torch 0.3.3

CRAN release: 2026-01-31

- Feat: Improve `lazy_tensor` printing.
- Fix: Improve consistency in
  [`as_lazy_tensor()`](https://mlr3torch.mlr-org.com/dev/reference/as_lazy_tensor.md)
  when converting 1D tensors to lazy tensors.
- Various minor bug fixes

## mlr3torch 0.3.2

CRAN release: 2025-10-31

### Bug Fixes

- `t_opt("adamw")` now actually uses AdamW and not Adam.
- Caching: Cache directory is now created, even if its parent directory
  does not exist.
- Add `mlr3torch` to `mlr_reflections$loaded_packages` to fix errors
  when using `mlr3torch` in parallel.

## mlr3torch 0.3.1

CRAN release: 2025-08-26

### Bug Fixes

- FT Transformer can now be (un-)marshaled after being trained on
  categorical data
  ([\#412](https://github.com/mlr-org/mlr3torch/issues/412)).
- Parameters (batch)-sampler now work
  ([\#420](https://github.com/mlr-org/mlr3torch/issues/420), thanks
  [@tdhock](https://github.com/tdhock))

### Features

- Better error messages.

## mlr3torch 0.3.0

CRAN release: 2025-07-07

### Breaking Changes:

- The output dimension of neural networks for binary classification
  tasks is now expected to be 1 and not 2 as before. The behavior of
  `nn("head")` was also changed to match this. This means that for
  binary classification tasks, `t_loss("cross_entropy")` now generates
  `nn_bce_with_logits_loss` instead of `nn_cross_entropy_loss`. This
  also came with a reparametrization of the `t_loss("cross_entropy")`
  loss (thanks to [@tdhock](https://github.com/tdhock),
  [\#374](https://github.com/mlr-org/mlr3torch/issues/374)).

### New Features:

#### PipeOps & Learners:

- Added `po("nn_identity")`
- Added `po("nn_fn")` for calling custom functions in a network.
- Added the FT Transformer model for tabular data.
- Added encoders for numericals and categoricals
- `nn("block")` (which allows to repeat the same network segment
  multiple times) now has an extra argument `trafo`, which allows to
  modify the parameter values per layer.

#### Callbacks:

- The context for callbacks now includes the network prediction
  (`y_hat`).
- The `lr_one_cycle` callback now infers the total number of steps.
- Progress callback got argument `digits` for controlling the precision
  with which validation/training scores are logged.

#### Other:

- `TorchIngressToken` now also can take a `Selector` as argument
  `features`.
- Added function
  [`lazy_shape()`](https://mlr3torch.mlr-org.com/dev/reference/lazy_shape.md)
  to get the shape of a lazy tensor.
- Better error messages for MLP and TabResNet learners.
- TabResNet learner now supports lazy tensors.
- The `LearnerTorch` base class now supports the private method
  `$.ingress_tokens(task, param_vals)` for generating the
  [`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html).
- Shapes can now have multiple `NA`s and not only the batch dimension
  can be missing. However, most
  [`nn()`](https://mlr3torch.mlr-org.com/dev/reference/nn.md) operators
  still expect only one missing values and will throw an error if
  multiple dimensions are unknown.
- Training now does not fail anymore when encountering a missing value
  during validation but uses `NA` instead.
- It is now possible to specify parameter groups for optimizers via the
  `param_groups` parameter.

### Bug Fixes:

- fix: lazy tensors of length 0 can now be materialized.
- fix: `NA` is now a valid shape for lazy tensors
- fix: The `lr_reduce_on_plateau` callback now works.

## mlr3torch 0.2.1

CRAN release: 2025-02-13

### Bug Fixes:

- `LearnerTorchModel` can now be parallelized and trained with
  encapsulation activated.
- `jit_trace` now works in combination with batch normalization.
- Ensures compatibility with `R6` version 2.6.0

## mlr3torch 0.2.0

CRAN release: 2025-02-07

### Breaking Changes

- Removed some optimizers for which no fast (‘ignite’) variant exists.
- The private `LearnerTorch$.dataloader()` method now operates no longer
  on the `task` but on the `dataset` generated by the private
  `LearnerTorch$.dataset()` method.
- The `shuffle` parameter during model training is now initialized to
  `TRUE` to sidestep issues where data is sorted.

### Performance Improvements

- Optimizers now use the faster (‘ignite’) version of the optimizers,
  which leads to considerable speed improvements.
- The `jit_trace` parameter was added to `LearnerTorch`, which when set
  to `TRUE` can lead to significant speedups. This should only be
  enabled for ‘static’ models, see the [torch
  tutorial](https://torch.mlverse.org/docs/articles/torchscript) for
  more information.
- Added parameter `num_interop_threads` to `LearnerTorch`.
- The `tensor_dataset` parameter was added, which allows to stack all
  batches at the beginning of training to make loading of batches
  afterwards faster.
- Use a faster default image loader.

### Features

- Added `PipeOp` for adaptive average pooling.
- The `n_layers` parameter was added to the MLP learner.
- Added multimodal melanoma and cifar{10, 100} example tasks.
- Added a callback to iteratively unfreeze parameters for finetuning.
- Added different learning rate schedulers as callbacks.

### Bug Fixes:

- Torch learners can now be used with `AutoTuner`.
- Early stopping now not uses `epochs - patience` for the internally
  tuned values instead of the trained number of `epochs` as it was
  before.
- The `dataset` of a learner must no longer return the tensors on the
  specified `device`, which allows for parallel dataloading on GPUs.
- `PipeOpBlock` should no longer create ID clashes with other PipeOps in
  the graph ([\#260](https://github.com/mlr-org/mlr3torch/issues/260)).

## mlr3torch 0.1.2

CRAN release: 2024-10-15

- Don’t use deprecated `data_formats` anymore
- Added `CallbackSetTB`, which allows logging that can be viewed by
  TensorBoard.

## mlr3torch 0.1.1

CRAN release: 2024-09-12

- fix(preprocessing): regarding the construction of some `PipeOps` such
  as `po("trafo_resize")` which failed in some cases.
- fix(ci): tests were not run in the CI
- fix(learner): `LearnerTabResnet` now works correctly
- Fix that tests were not run in the CI
- feat: added the
  [`nn()`](https://mlr3torch.mlr-org.com/dev/reference/nn.md) helper
  function to simplify the creation of neural network layers

## mlr3torch 0.1.0

CRAN release: 2024-07-08

- Initial CRAN release
