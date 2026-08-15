# mlr3torch (development version)

## Breaking changes

* The `freq_type` parameter of `t_clbk("checkpoint")` was removed and checkpointing is always per epoch.
* The construction argument `only_batch_unknown` of `PipeOpTorch` was removed,
  as shape inference functions are now expected to handle multiple unknown
  dimensions.
* The dropout probability `p` of `lrn("classif.mlp")` / `lrn("regr.mlp")` is now initialized to
  `0.1` instead of `0.5`. Set `p = 0.5` explicitly to keep the old behaviour.

## Features

* Added more image learners from {torchvision}.
* `LearnerTorch` now tracks the validation scores of the best epoch and exposes them via the new
  `$best_valid_scores` field, so `msr("best_valid_score")` can be used for tuning.
  This is the epoch that is also reported via `$internal_tuned_values`, whereas `$internal_valid_scores`
  refers to the last epoch, i.e. to the network that is actually returned.
  Tracking requires early stopping to be active (`patience > 0`).
* Most `LearnerTorchVision` are now `jittable`.
* Ported the `TabM` tabular learner from Python.
* `LearnerTorch` now has `.loss_fn(task, param_vals)` private method that allows
  to customize the construction of the loss function.
* `LearnerTorch` now has `restore_best_weights` parameter that can be used when
   early stopping is active.
* A network can now return more than one prediction during training as a list.
  The first is expected to be the primary prediction.
  In `ContextTorch`, `$y_hat` is the primary prediction and `$y_hats` contains
  the complete prediction.
* New parameter `batch_size_predict` for `LearnerTorch`, which overrides `batch_size` for prediction
* Added multihead attention and transformer encoder pipeops.
* Any dimension of an input shape can now be unknown (`NA`), not only the batch dimension.
* Improved error messages during `PipeOpTorch`'s shape inference.
* The `shape` parameter of `nn("reshape")` can now be a `function(shape)` of the input shape.
* Exported various helpers useful for implementing shape inference for custom `PipeOpTorch` classes.
* `ContextTorch` has a new field `$callbacks`, which gives a callback access to the other callbacks
  of the training run.
* `CallbackSet` has a new field `$weight` that controls when a callback is called within a stage.
* `t_clbk("checkpoint")` now accepts an existing empty directory as its `path`

## Bug fixes

* `ContextTorch$epoch` is now `0` during the `on_begin` stage instead of `NULL`.
* `t_clbk("checkpoint")` no longer writes an epoch that was interrupted
  under that epoch's own number, so `network<n>.pt` is now always the
  network at the *end* of epoch `n` rather than sometimes a half-trained one.
* `replace_head()` for `mobilenet_v2` and `VGG` works for `width_mult` above 1.
* `PipeOpTorch$shapes_out()` now always returns `integer()` shapes (and not
    sometimes doubles like `NA`).
* `po("torch_model_classif")` and `po("torch_model_regr")` now have the correct
  `$packages`.
* The `batch_sampler` parameter can now be used without setting `batch_size` for training.
* Configuration errors that are only caught during `LearnerTorch` no longer
  trigger a fallback learner.
* The `LearnerTorch`'s `sampler` and `batch_sampler` parameters are now not used
  during prediction.
* `logical()` features are now encoded as 1-based instead of 1-based.
* `lazy_tensor` columns are now again printed correctly inside `data.table`s
* Fixed some links on the pkgdown website and the help pages.
* Fixed various other shape inference bugs.
* `po("torch_model_{regr, classif}")` now resets the parameters of the network
  at the beginning of `$train()` when the network is built from `PipeOpTorch` objects,
  which makes the results reproducible for the set `seed` parameter.
* `nn()` now properly interprets `nn("linear_1")` as `po("nn_linear", id = "linear_21")`.
* Fixed some bugs in `FTTransformer`: `attention_initialization` now has an
  effect, `n_blocks = 0` is allowed and the hidden dimension falls back to
  `d_token * 4/3` as in the reference implementation.
* Fixed some issues in the documentation.

# mlr3torch 0.3.3

* Feat: Improve `lazy_tensor` printing.
* Fix: Improve consistency in `as_lazy_tensor()` when converting 1D tensors to lazy tensors.
* Various minor bug fixes

# mlr3torch 0.3.2

## Bug Fixes

* `t_opt("adamw")` now actually uses AdamW and not Adam.
* Caching: Cache directory is now created, even if its parent
  directory does not exist.
* Add `mlr3torch` to `mlr_reflections$loaded_packages` to fix errors when using `mlr3torch` in parallel.

# mlr3torch 0.3.1

## Bug Fixes

* FT Transformer can now be (un-)marshaled after being trained on categorical data (#412).
* Parameters (batch)-sampler now work (#420, thanks @tdhock)

## Features

* Better error messages.

# mlr3torch 0.3.0

## Breaking Changes:

* The output dimension of neural networks for binary classification tasks is now
  expected to be 1 and not 2 as before. The behavior of `nn("head")` was also changed to match this.
  This means that for binary classification tasks, `t_loss("cross_entropy")` now generates
  `nn_bce_with_logits_loss` instead of `nn_cross_entropy_loss`.
  This also came with a reparametrization of the `t_loss("cross_entropy")` loss (thanks to @tdhock, #374).

## New Features:


### PipeOps & Learners:

* Added `po("nn_identity")`
* Added `po("nn_fn")` for calling custom functions in a network.
* Added the FT Transformer model for tabular data.
* Added encoders for numericals and categoricals
* `nn("block")` (which allows to repeat the same network segment multiple
  times) now has an extra argument `trafo`, which allows to modify the
  parameter values per layer.

### Callbacks:

* The context for callbacks now includes the network prediction (`y_hat`).
* The `lr_one_cycle` callback now infers the total number of steps.
* Progress callback got argument `digits` for controlling the precision
  with which validation/training scores are logged.

### Other:

* `TorchIngressToken` now also can take a `Selector` as argument `features`.
* Added function `lazy_shape()` to get the shape of a lazy tensor.
* Better error messages for MLP and TabResNet learners.
* TabResNet learner now supports lazy tensors.
* The `LearnerTorch` base class now supports the private method `$.ingress_tokens(task, param_vals)`
  for generating the `torch::dataset`.
* Shapes can now have multiple `NA`s and not only the batch dimension can be missing. However, most `nn()` operators
  still expect only one missing values and will throw an error if multiple dimensions are unknown.
* Training now does not fail anymore when encountering a missing value
  during validation but uses `NA` instead.
* It is now possible to specify parameter groups for optimizers via the
`param_groups` parameter.


## Bug Fixes:

* fix: lazy tensors of length 0 can now be materialized.
* fix: `NA` is now a valid shape for lazy tensors
* fix: The `lr_reduce_on_plateau` callback now works.

# mlr3torch 0.2.1

## Bug Fixes:

* `LearnerTorchModel` can now be parallelized and trained with
  encapsulation activated.
* `jit_trace` now works in combination with batch normalization.
* Ensures compatibility with `R6` version 2.6.0

# mlr3torch 0.2.0

## Breaking Changes

* Removed some optimizers for which no fast ('ignite') variant exists.
* The private `LearnerTorch$.dataloader()` method now operates no longer
  on the `task` but on the `dataset` generated by the private `LearnerTorch$.dataset()` method.
* The `shuffle` parameter during model training is now initialized to `TRUE` to sidestep
  issues where data is sorted.

## Performance Improvements

* Optimizers now use the faster ('ignite') version of the optimizers,
  which leads to considerable speed improvements.
* The `jit_trace` parameter was added to `LearnerTorch`, which when set to
  `TRUE` can lead to significant speedups.
  This should only be enabled for 'static' models, see the
  [torch tutorial](https://torch.mlverse.org/docs/articles/torchscript)
  for more information.
* Added parameter `num_interop_threads` to `LearnerTorch`.
* The `tensor_dataset` parameter was added, which allows to stack all batches
  at the beginning of training to make loading of batches afterwards faster.
* Use a faster default image loader.

## Features

* Added `PipeOp` for adaptive average pooling.
* The `n_layers` parameter was added to the MLP learner.
* Added multimodal melanoma and cifar{10, 100} example tasks.
* Added a callback to iteratively unfreeze parameters for finetuning.
* Added different learning rate schedulers as callbacks.

## Bug Fixes:

* Torch learners can now be used with `AutoTuner`.
* Early stopping now not uses `epochs - patience` for the internally tuned
  values instead of the trained number of `epochs` as it was before.
* The `dataset` of a learner must no longer return the tensors on the specified `device`,
  which allows for parallel dataloading on GPUs.
* `PipeOpBlock` should no longer create ID clashes with other PipeOps in the graph (#260).

# mlr3torch 0.1.2

* Don't use deprecated `data_formats` anymore
* Added `CallbackSetTB`, which allows logging that can be viewed by TensorBoard.

# mlr3torch 0.1.1

* fix(preprocessing): regarding the construction of some `PipeOps` such as `po("trafo_resize")`
  which failed in some cases.
* fix(ci): tests were not run in the CI
* fix(learner): `LearnerTabResnet` now works correctly
* Fix that tests were not run in the CI
* feat: added the `nn()` helper function to simplify the creation of neural network
  layers

# mlr3torch 0.1.0

* Initial CRAN release
