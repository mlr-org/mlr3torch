# PipeOp Torch Model

Builds a Torch Learner from a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
and trains it with the given parameter specification. The task type must
be specified during construction.

## Parameters

**General**:

The parameters of the optimizer, loss and callbacks, prefixed with
`"opt."`, `"loss."` and `"cb.<callback id>."` respectively, as well as:

- `epochs` :: `integer(1)`  
  The number of epochs.

- `device` :: `character(1)`  
  The device. One of `"auto"`, `"cpu"`, or `"cuda"` or other values
  defined in `mlr_reflections$torch$devices`. The value is initialized
  to `"auto"`, which will select `"cuda"` if possible, then try `"mps"`
  and otherwise fall back to `"cpu"`.

- `num_threads` :: `integer(1)`  
  The number of threads for intraop pararallelization (if `device` is
  `"cpu"`). This value is initialized to 1.

- `num_interop_threads` :: `integer(1)`  
  The number of threads for intraop and interop pararallelization (if
  `device` is `"cpu"`). This value is initialized to 1. Note that this
  can only be set once during a session and changing the value within an
  R session will raise a warning.

- `seed` :: `integer(1)` or `"random"` or `NULL`  
  The torch seed that is used during training and prediction. This value
  is initialized to `"random"`, which means that a random seed will be
  sampled at the beginning of the training phase. This seed (either set
  or randomly sampled) is available via `$model$seed` after training and
  used during prediction. Note that by setting the seed during the
  training phase this will mean that by default (i.e. when `seed` is
  `"random"`), clones of the learner will use a different seed. If set
  to `NULL`, no seeding will be done.

- `tensor_dataset` :: `logical(1)` \| `"device"`  
  Whether to load all batches at once at the beginning of training and
  stack them. This is initialized to `FALSE`. If set to `"device"`, the
  device of the tensors will be set to the value of `device`, which can
  avoid unnecessary moving of tensors between devices. When your dataset
  fits into memory this will make the loading of batches faster. Note
  that this should not be set for datasets that contain
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)s
  with random data augmentation, as this augmentation will only be
  applied once at the beginning of training.

**Evaluation**:

- `measures_train` ::
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) or
  [`list()`](https://rdrr.io/r/base/list.html) of
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html)s  
  Measures to be evaluated during training.

- `measures_valid` ::
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) or
  [`list()`](https://rdrr.io/r/base/list.html) of
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html)s  
  Measures to be evaluated during validation.

- `eval_freq` :: `integer(1)`  
  How often the train / validation predictions are evaluated using
  `measures_train` / `measures_valid`. This is initialized to `1`. Note
  that the final model is always evaluated.

**Early Stopping**:

- `patience` :: `integer(1)`  
  This activates early stopping using the validation scores. If the
  performance of a model does not improve for `patience` evaluation
  steps, training is ended. Note that the final model is stored in the
  learner, not the best model. This is initialized to `0`, which means
  no early stopping. The first entry from `measures_valid` is used as
  the metric. This also requires to specify the `$validate` field of the
  Learner, as well as `measures_valid`. If this is set, the epoch after
  which no improvement was observed, can be accessed via the
  `$internal_tuned_values` field of the learner.

- `min_delta` :: `double(1)`  
  The minimum improvement threshold for early stopping. Is initialized
  to 0.

**Dataloader**:

- `batch_size` :: `integer(1)`  
  The batch size (required).

- `shuffle` :: `logical(1)`  
  Whether to shuffle the instances in the dataset. This is initialized
  to `TRUE`, which differs from the default (`FALSE`).

- `sampler` ::
  [`torch::sampler`](https://torch.mlverse.org/docs/reference/sampler.html)  
  Object that defines how the dataloader draw samples.

- `batch_sampler` ::
  [`torch::sampler`](https://torch.mlverse.org/docs/reference/sampler.html)  
  Object that defines how the dataloader draws batches.

- `num_workers` :: `integer(1)`  
  The number of workers for data loading (batches are loaded in
  parallel). The default is `0`, which means that data will be loaded in
  the main process.

- `collate_fn` :: `function`  
  How to merge a list of samples to form a batch.

- `pin_memory` :: `logical(1)`  
  Whether the dataloader copies tensors into CUDA pinned memory before
  returning them.

- `drop_last` :: `logical(1)`  
  Whether to drop the last training batch in each epoch during training.
  Default is `FALSE`.

- `timeout` :: `numeric(1)`  
  The timeout value for collecting a batch from workers. Negative values
  mean no timeout and the default is `-1`.

- `worker_init_fn` :: `function(id)`  
  A function that receives the worker id (in `[1, num_workers]`) and is
  exectued after seeding on the worker but before data loading.

- `worker_globals` :: [`list()`](https://rdrr.io/r/base/list.html) \|
  [`character()`](https://rdrr.io/r/base/character.html)  
  When loading data in parallel, this allows to export globals to the
  workers. If this is a character vector, the objects in the global
  environment with those names are copied to the workers.

- `worker_packages` ::
  [`character()`](https://rdrr.io/r/base/character.html)  
  Which packages to load on the workers.

Also see `torch::dataloder` for more information.

## Input and Output Channels

There is one input channel `"input"` that takes in `ModelDescriptor`
during traing and a `Task` of the specified `task_type` during
prediction. The output is `NULL` during training and a `Prediction` of
given `task_type` during prediction.

## State

A trained
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md).

## Internals

A
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)
is created by calling
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md)
on the provided
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
that is received through the input channel. Then the parameters are set
according to the parameters specified in `PipeOpTorchModel` and its
'\$train()` method is called on the [`Task`][mlr3::Task] stored in the [`ModelDescriptor\`\].

## See also

Other PipeOps:
[`mlr_pipeops_nn_adaptive_avg_pool1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_adaptive_avg_pool1d.md),
[`mlr_pipeops_nn_adaptive_avg_pool2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_adaptive_avg_pool2d.md),
[`mlr_pipeops_nn_adaptive_avg_pool3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_adaptive_avg_pool3d.md),
[`mlr_pipeops_nn_avg_pool1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_avg_pool1d.md),
[`mlr_pipeops_nn_avg_pool2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_avg_pool2d.md),
[`mlr_pipeops_nn_avg_pool3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_avg_pool3d.md),
[`mlr_pipeops_nn_batch_norm1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_batch_norm1d.md),
[`mlr_pipeops_nn_batch_norm2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_batch_norm2d.md),
[`mlr_pipeops_nn_batch_norm3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_batch_norm3d.md),
[`mlr_pipeops_nn_block`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_block.md),
[`mlr_pipeops_nn_celu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_celu.md),
[`mlr_pipeops_nn_conv1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv1d.md),
[`mlr_pipeops_nn_conv2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv2d.md),
[`mlr_pipeops_nn_conv3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv3d.md),
[`mlr_pipeops_nn_conv_transpose1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv_transpose1d.md),
[`mlr_pipeops_nn_conv_transpose2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv_transpose2d.md),
[`mlr_pipeops_nn_conv_transpose3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv_transpose3d.md),
[`mlr_pipeops_nn_dropout`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_dropout.md),
[`mlr_pipeops_nn_elu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_elu.md),
[`mlr_pipeops_nn_flatten`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_flatten.md),
[`mlr_pipeops_nn_ft_cls`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_ft_cls.md),
[`mlr_pipeops_nn_ft_transformer_block`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_ft_transformer_block.md),
[`mlr_pipeops_nn_geglu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_geglu.md),
[`mlr_pipeops_nn_gelu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_gelu.md),
[`mlr_pipeops_nn_glu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_glu.md),
[`mlr_pipeops_nn_hardshrink`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_hardshrink.md),
[`mlr_pipeops_nn_hardsigmoid`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_hardsigmoid.md),
[`mlr_pipeops_nn_hardtanh`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_hardtanh.md),
[`mlr_pipeops_nn_head`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md),
[`mlr_pipeops_nn_identity`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_identity.md),
[`mlr_pipeops_nn_layer_norm`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_layer_norm.md),
[`mlr_pipeops_nn_leaky_relu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_leaky_relu.md),
[`mlr_pipeops_nn_linear`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_linear.md),
[`mlr_pipeops_nn_log_sigmoid`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_log_sigmoid.md),
[`mlr_pipeops_nn_max_pool1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_max_pool1d.md),
[`mlr_pipeops_nn_max_pool2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_max_pool2d.md),
[`mlr_pipeops_nn_max_pool3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_max_pool3d.md),
[`mlr_pipeops_nn_merge`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge.md),
[`mlr_pipeops_nn_merge_cat`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_cat.md),
[`mlr_pipeops_nn_merge_prod`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_prod.md),
[`mlr_pipeops_nn_merge_sum`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_sum.md),
[`mlr_pipeops_nn_prelu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_prelu.md),
[`mlr_pipeops_nn_reglu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_reglu.md),
[`mlr_pipeops_nn_relu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_relu.md),
[`mlr_pipeops_nn_relu6`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_relu6.md),
[`mlr_pipeops_nn_reshape`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_reshape.md),
[`mlr_pipeops_nn_rrelu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_rrelu.md),
[`mlr_pipeops_nn_selu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_selu.md),
[`mlr_pipeops_nn_sigmoid`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_sigmoid.md),
[`mlr_pipeops_nn_softmax`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softmax.md),
[`mlr_pipeops_nn_softplus`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softplus.md),
[`mlr_pipeops_nn_softshrink`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softshrink.md),
[`mlr_pipeops_nn_softsign`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softsign.md),
[`mlr_pipeops_nn_squeeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_squeeze.md),
[`mlr_pipeops_nn_tanh`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tanh.md),
[`mlr_pipeops_nn_tanhshrink`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tanhshrink.md),
[`mlr_pipeops_nn_threshold`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_threshold.md),
[`mlr_pipeops_nn_tokenizer_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tokenizer_categ.md),
[`mlr_pipeops_nn_tokenizer_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tokenizer_num.md),
[`mlr_pipeops_nn_unsqueeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_unsqueeze.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_model_classif`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model_classif.md),
[`mlr_pipeops_torch_model_regr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model_regr.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3pipelines::PipeOpLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_learner.html)
-\> `PipeOpTorchModel`

## Methods

### Public methods

- [`PipeOpTorchModel$new()`](#method-PipeOpTorchModel-new)

- [`PipeOpTorchModel$clone()`](#method-PipeOpTorchModel-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchModel$new(task_type, id = "torch_model", param_vals = list())

#### Arguments

- `task_type`:

  (`character(1)`)  
  The task type of the model.

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchModel$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
