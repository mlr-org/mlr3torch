# mlr3torch (development version)

## Features

* Added learners for the remaining image classification networks of `torchvision`:
  ConvNeXt (`classif.convnext_*`), EfficientNet (`classif.efficientnet_b0` to
  `classif.efficientnet_b7`), EfficientNetV2 (`classif.efficientnet_v2_{s,m,l}`),
  Inception v3 (`classif.inception_v3`), MaxViT (`classif.maxvit`), MobileNetV3
  (`classif.mobilenet_v3_{large,small}`), Vision Transformers (`classif.vit_*`) and
  Wide ResNet (`classif.wide_resnet{50_2,101_2}`).
* `LearnerTorch` gained a private `.loss_fn(task, param_vals)` method, which constructs the loss
  that is applied to the output of the network and by default returns `self$loss$generate(task)`.
  Learners can overload it to wrap the loss that was configured by the user, instead of the loss
  being generated inline in the training loop.
* A network can now return more than one prediction during training: it may return a `list()` of
  tensors, where the first element is the primary prediction that is scored by `measures_train` and
  returned when predicting, and the remaining elements are the predictions of auxiliary classifiers
  that only contribute to the loss.
  This is documented in the "Network Head and Target Encoding" section of `LearnerTorch`.
* `ContextTorch` gained the field `y_hats`, which holds the complete output of the network for the
  current batch, i.e. what the loss is applied to. `y_hat` now always holds the *primary*
  prediction, so callbacks that read it keep working for networks with auxiliary classifiers.
  For a network that returns a single tensor the two are identical.
* Added the `TabM` learner (`lrn("classif.tabm")` / `lrn("regr.tabm")`), a port of the
  official TabM reference implementation. Numerical features can optionally be embedded via the
  `num_embeddings` parameter, which supports the linear-ReLU, periodic and piecewise-linear
  embeddings of the `rtdl_num_embeddings` package.
* New parameter `batch_size_predict` for `LearnerTorch`, which overrides `batch_size` for prediction
  (including the validation data during training) when it is set.
* The `batch_sampler` parameter can now be used without setting `batch_size` for training,
  as the batch sampler already determines the batches (#420).
  A `batch_size` (or `batch_size_predict`) is still required for prediction, where the (batch)
  sampler is not used.
* The dataloader parameters are now validated with typed conditions: misconfigurations are signaled
  as `Mlr3ErrorConfig` (see `mlr3misc::error_config()`), so they can be caught by class and do not
  trigger the fallback learner anymore.
* Added `PipeOpTorchMultiheadAttention` (`po("nn_multihead_attention")`), which wraps
  `torch::nn_multihead_attention()`.
* Many `PipeOp`s now accept input shapes in which dimensions other than the batch
  dimension are unknown (`NA`). This affects operators that never inspect the unknown
  dimension when building their module: the activation functions, `nn_dropout`,
  `nn_identity`, `nn_softmax`, the `nn_merge_*` operators, all pooling operators and the
  reshaping operators. Previously only the batch dimension was allowed to be unknown.
  Notably, `nn_adaptive_avg_pool*` can now resolve an unknown input extent to a fully
  known output shape.
* `PipeOp`s that read only *some* dimensions of the input shape now require only those
  dimensions to be known: `nn_conv*` and `nn_conv_transpose*` need the channel dimension,
  `nn_batch_norm*` the feature dimension, `nn_layer_norm` the last `dims` dimensions,
  `nn_ft_cls` the token dimension and `nn_squeeze` the squeezed dimension. Convolutional
  networks can therefore now be built for images whose height and width are unknown.
  `nn_fn` and `nn_block` accept unknown dimensions whenever the wrapped function or the
  wrapped `PipeOp`s do. When a required dimension is unknown, the resulting error message
  now names that dimension instead of failing inside `libtorch`.
* All preprocessing `PipeOp`s now compute their output shapes exactly instead of inferring
  them by tracing the wrapped function. `infer_shapes()` is now only used for operators that the
  user supplies (`nn_fn` without `shapes_out`, `pipeop_preproc_torch()` with
  `shapes_out = "infer"`). The shape rules are verified against the shape that the wrapped
  function actually returns, computed on torch's `"meta"` device.
* `infer_shapes()` now traces the unknown dimensions with a spread of values instead of `1`,
  `2` and `3`, and drops a trace that fails. Filling in `1` made operators such as
  `torch_squeeze()` change the number of output dimensions and rejected valid shapes, and small
  values reported a wrong output shape for operators that clamp to the input size. A trace that
  needs a larger extent (a convolution with a large kernel) no longer fails.
* `nn_reshape` resolves an unknown dimension of `shape` when the number of input elements is
  known, e.g. an input shape of `(32, 4, 6)` with `shape = c(-1, 24)` now gives `(32, 24)`.
* `nn_squeeze` without a `dim` no longer requires all dimensions to be known. Unknown
  dimensions are assumed to not be 1 and are kept, and the module squeezes exactly those
  dimensions that `$shapes_out()` squeezed.

## Breaking changes

* The construction argument `only_batch_unknown` of `PipeOpTorch` was removed.
  Any dimension of an input shape can now be unknown, so `private$.shapes_out()` must always
  handle `NA`s and assert those dimensions it actually needs to be known.

## Bug fixes

* The `sampler` and `batch_sampler` parameters are no longer used during prediction, where they
  could silently misalign the predictions with the rows of the task.
  They are now tagged with `"train"` only.
* `logical()` features are now encoded as `c(1, 2)` by the
`batchgetter_categ()` and their cardinality is correctly computed.
* `lazy_tensor` columns are now again printed correctly inside `data.table`s
* `LearnerTorchVision` did not pass its `jittable` argument on to its parent class, so none
  of the `torchvision` learners had a `jit_trace` parameter and none of them could be traced.
  `jit_trace` is now available for all of them except the vision transformers, whose attention
  blocks cannot be traced, and Inception v3, whose auxiliary classifier makes the network return
  more than one prediction.
* `nn_layer_norm` could not be used with `dims > 1`, because the parameter was checked
  against the number of input channels instead of the number of input dimensions.
* `trafo_resize` reported a square output for a `size` of length 1, although
  `torchvision::transform_resize()` matches the *shorter* side and preserves the aspect ratio.
* `nn_max_pool1d`, `nn_max_pool2d` and `nn_max_pool3d` ignored the `dilation` parameter when
  computing their output shape, and all pooling operators ignored that torch drops a pooling
  window which would start inside the right-hand padding when `ceil_mode` is `TRUE`. Both reported
  wrong output shapes for fully known input shapes.
* `nn_ft_transformer_block` reported a single output token when `query_idx` was set, although
  it returns one token per queried index.
* `nn_squeeze` rejected a `dim` of length greater than 1, although the module supports it.
* `nn_merge_cat` broadcast the dimensions it does not concatenate, e.g. it accepted the input
  shapes `(NA, 4, 1)` and `(NA, 4, 6)`. `torch_cat()` requires those dimensions to be equal, so
  such shapes are now rejected when the network is built instead of failing at runtime.
* `nn_glu` *extended* the input shape with unknown dimensions when `dim` was outside the range
  of the input, so a network could be built on an impossible shape. The `dim` is now checked
  against the number of input dimensions, and negative values are supported.
* The convolution operators matched their `padding` parameter partially, so setting
  `padding_mode` without `padding` used the padding mode as the padding.
* `nn_squeeze` with a negative `dim` squeezed different dimensions than the module it built.
* `nn_reshape` silently changed the batch size when `shape` began with a fixed value. The
  batch dimension is now only allowed to change when it is given as `-1`.
* `nn_tokenizer_categ` took the number of tokens from the input shape instead of the
  cardinalities, passed `cardinalities` twice to its module and did not reject tasks without
  categorical features or with a mismatching number of features.
* `nn_ft_transformer_block` contained an unreachable postnormalization branch and did not
  check `query_idx` against the number of tokens.
* The `initialization` parameter of `nn_ft_cls` never reached the module, because it was
  declared as a default instead of an initial value.
* `nn_block` failed for a known batch size and for `n_blocks = 0`.
* `nn_head` failed with an uninformative error when it was used without a task.
* The preprocessing operators are computed at the *predict* stage from the predict input
  shape instead of the train input shape. Pipelines whose shapes differ between the two stages
  previously trained successfully and then failed at predict time.
* `trafo_adjust_hue` truncated its input to three channels, and the flipping operators
  required RGB input although they work for any number of channels.
* `nn_softshrink` swapped the bounds of its `lambd` parameter, and `nn_rrelu` used wrong
  defaults for `lower` and `upper`. Required vector-valued parameters such as `kernel_size` no
  longer accept `NULL`.

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
* The default optimizer is now AdamW instead of Adam.
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
