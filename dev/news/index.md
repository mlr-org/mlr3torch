# Changelog

## mlr3torch (development version)

### Features

- Added learners for the remaining image classification networks of
  `torchvision`: ConvNeXt (`classif.convnext_*`), EfficientNet
  (`classif.efficientnet_b0` to `classif.efficientnet_b7`),
  EfficientNetV2 (`classif.efficientnet_v2_{s,m,l}`), Inception v3
  (`classif.inception_v3`), MaxViT (`classif.maxvit`), MobileNetV3
  (`classif.mobilenet_v3_{large,small}`), Vision Transformers
  (`classif.vit_*`) and Wide ResNet (`classif.wide_resnet{50_2,101_2}`).
- `LearnerTorch` gained a private `.loss_fn(task, param_vals)` method,
  which constructs the loss that is applied to the output of the network
  and by default returns `self$loss$generate(task)`. Learners can
  overload it to wrap the loss that was configured by the user, instead
  of the loss being generated inline in the training loop.
- A network can now return more than one prediction during training: it
  may return a [`list()`](https://rdrr.io/r/base/list.html) of tensors,
  where the first element is the primary prediction that is scored by
  `measures_train` and returned when predicting, and the remaining
  elements are the predictions of auxiliary classifiers that only
  contribute to the loss. In `ContextTorch`, the list of predictions
  available as `y_hats`, while `y_hat` now refers to the first
  prediction. This is documented in the “Network Head and Target
  Encoding” section of `LearnerTorch`.
- Added the `TabM` learner (`lrn("classif.tabm")` / `lrn("regr.tabm")`),
  a port of the official TabM reference implementation.
- New parameter `batch_size_predict` for `LearnerTorch`, which overrides
  `batch_size` for prediction (including the validation data during
  training) when it is set.
- Added `PipeOpTorchMultiheadAttention`
  (`po("nn_multihead_attention")`).
- Most `LearnerTorchVision` are now `jittable`.
- Any dimension of an input shape can now be unknown (`NA`), not only
  the batch dimension.
- Improved error messages during `PipeOpTorch`’s shape inference.
- The `shape` parameter of `nn("reshape")` can now be a
  `function(shape)` of the input shape.
- Exported various helpers useful for implementing shape inference for
  custom `PipeOpTorch` classes.
- [`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
  now accepts a known batch dimension in `pointer_shape`, so an operator
  can check what it would otherwise have to assume, e.g. that a reshape
  keeps the batch dimension.

### Breaking changes

- The `freq_type` parameter of `t_clbk("checkpoint")` was removed;
  checkpoints are now always written per epoch. `freq_type = "step"`
  named its files after the within-epoch step, which restarts at every
  epoch, so each epoch silently overwrote the checkpoints of the
  previous one. Code that set `freq_type` – including to its default
  `"epoch"` – has to drop the argument.
- The construction argument `only_batch_unknown` of `PipeOpTorch` was
  removed. Any dimension of an input shape can now be unknown, so
  `private$.shapes_out()` must always handle `NA`s and assert those
  dimensions it actually needs to be known.

### Bug fixes

- `t_clbk("checkpoint")` no longer writes an epoch that was interrupted
  – because training failed or was stopped early – under that epoch’s
  own number, so `network<n>.pt` is now always the network at the *end*
  of epoch `n` rather than sometimes a half-trained one.
- `t_clbk("checkpoint")` now accepts an existing empty directory as its
  `path`. Previously any existing directory was rejected, which made a
  pre-created output folder unusable and meant that a run failing before
  its first checkpoint left behind a folder that blocked every later
  run.
- The `batch_sampler` parameter can now be used without setting
  `batch_size` for training, as the batch sampler already determines the
  batches ([\#420](https://github.com/mlr-org/mlr3torch/issues/420)).
- The dataloader parameters are now validated with typed conditions:
  misconfigurations are signaled as `Mlr3ErrorConfig` (see
  [`mlr3misc::error_config()`](https://mlr3misc.mlr-org.com/reference/mlr_conditions.html)),
  so they do not trigger the fallback learner anymore.
- [`nn()`](https://mlr3torch.mlr-org.com/dev/reference/nn.md) now
  accepts a `_<n>` suffix on the key to disambiguate repeated layers
  within a `Graph`, i.e. `nn("linear_1")` is short for
  `nn("linear", id = "linear_1")`. Previously the suffix was appended a
  second time, resulting in the id `"linear_1_1"`.
- The `sampler` and `batch_sampler` parameters are no longer used during
  prediction, where they could silently misalign the predictions with
  the rows of the task. They are now tagged with `"train"` only.
- [`logical()`](https://rdrr.io/r/base/logical.html) features are now
  encoded as `c(1, 2)` by the
  [`batchgetter_categ()`](https://mlr3torch.mlr-org.com/dev/reference/batchgetter_categ.md)
  and their cardinality is correctly computed.
- `lazy_tensor` columns are now again printed correctly inside
  `data.table`s
- The callback overview on the package website now links to the correct
  help pages.
- `t_clbk("lr_one_cycle")` and `t_clbk("lr_reduce_on_plateau")` now
  point to their own help pages instead of the generic
  `mlr_callback_set.lr_scheduler` page.
- `nn("reshape")` with a `function(shape)` target now resolves a `-1`
  whenever the number of elements per observation is known, i.e. when
  the batch dimension is the only unknown one.
- Fixed various other shape inference bugs.

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
- The default optimizer is now AdamW instead of Adam.
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
