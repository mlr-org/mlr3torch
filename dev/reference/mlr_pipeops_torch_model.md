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
  The number of threads for intraop parallelization (if `device` is
  `"cpu"`). This value is initialized to 1. When resampling,
  benchmarking or tuning in parallel, each worker uses this many
  threads, so divide the available cores among the workers instead of
  setting this to the number of cores.

- `num_interop_threads` :: `integer(1)`  
  The number of threads for interop parallelization (if `device` is
  `"cpu"`). Note that this can only be set **once** per session, so
  setting this for one learner also changes the behavior of other
  learners, and a later learner asking for a different value errors.
  `NULL` (default) uses whatever is set. In order to use different
  values for this parameter, use encapsulation to train the learners in
  separate R sessions.

- `seed` :: `integer(1)` or `"random"` or `NULL`  
  The torch seed that is used during training and prediction. This value
  is initialized to `"random"`, which means that a random seed will be
  sampled at the beginning of the training phase. This seed (either set
  or randomly sampled) is available via `$model$seed` after training and
  used during prediction. Note that by setting the seed during the
  training phase this will mean that by default (i.e. when `seed` is
  `"random"`), clones of the learner will use a different seed. If set
  to `NULL`, no seeding will be done. This parameter only seeds torch's
  random number generator, it does **not** seed R's. Anything that is
  drawn from R's RNG is therefore unaffected by it, so to make those
  parts reproducible you need to seed R's RNG as well, e.g. via
  [`set.seed()`](https://rdrr.io/r/base/Random.html).

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

**Resuming**:

- `resume` :: `character(1)` or `TRUE`  
  Continues training from a checkpoint written by
  [`t_clbk("checkpoint")`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md),
  either the folder it wrote to or `TRUE`, which takes that folder from
  the checkpoint callback of this learner. Note that `epochs` is the
  *total* number of epochs, i.e. it includes the epochs the checkpoint
  was already trained for: resuming a checkpoint from epoch 5 with
  `epochs = 8` trains 3 more epochs.

**Early Stopping**:

- `patience` :: `integer(1)`  
  This activates early stopping using the validation scores. If the
  performance of a model does not improve for `patience` evaluation
  steps, training is ended. Note that this counts *evaluation steps*,
  not epochs: when `eval_freq` is greater than `1`, `patience`
  evaluation steps correspond to `patience * eval_freq` epochs. Note
  that the final model is stored in the learner, not the best model.
  This is initialized to `0`, which means no early stopping. The first
  entry from `measures_valid` is used as the metric. This also requires
  to specify the `$validate` field of the Learner, as well as
  `measures_valid`. If this is set, the epoch after which no improvement
  was observed, can be accessed via the `$internal_tuned_values` field
  of the learner.

- `min_delta` :: `double(1)`  
  The minimum improvement threshold for early stopping. Is initialized
  to 0.

- `restore_best_weights` :: `logical(1)`  
  Whether to restore the weights of the best epoch when training ends,
  instead of keeping those of the last epoch that was trained. Is
  initialized to `FALSE`, i.e. the network of the last epoch is stored.
  Setting this to `TRUE` makes the stored network the one of the epoch
  that `$internal_tuned_values` reports, and costs one additional copy
  of the network's parameters in memory. Checkpoints written by
  `t_clbk("checkpoint")` are unaffected: they always hold the network as
  training left it.

  When a run is resumed (see the section *Resuming* of
  [`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)),
  the best score, the epoch it was observed in and the number of
  evaluation steps without improvement are restored, so `patience` keeps
  counting across runs instead of starting over. The best epoch's
  weights are not part of a checkpoint, however – they are a full copy
  of the network, which every checkpoint would otherwise carry. A
  resumed run with `restore_best_weights` that never beats the restored
  best score therefore ends with the weights of its last epoch while
  `$internal_tuned_values` still reports the earlier best epoch, which
  is warned about when the state is restored.

**Dataloader**:

- `batch_size` :: `integer(1)`  
  The batch size used by the training and prediction dataloader. It is
  required for training (unless a `batch_sampler` is provided, which
  already determines the batches) and it is required for prediction
  (unless `batch_size_predict` is set).

- `batch_size_predict` :: `integer(1)`  
  The batch size used by the prediction dataloader (this includes the
  validation data during training). When set, it overrides `batch_size`
  for prediction. The batch size does not change the predictions, but
  smaller batches take longer and require less memory.

- `shuffle` :: `logical(1)`  
  Whether to shuffle the instances in the dataset. This is initialized
  to `TRUE`, which differs from the default (`FALSE`). It is ignored
  when a `sampler` or `batch_sampler` is provided.

- `sampler` ::
  [`torch::sampler`](https://torch.mlverse.org/docs/reference/sampler.html)  
  Object that defines how the dataloader draws samples, i.e. the order
  in which the observations are drawn. This must be the sampler
  *generator* (as returned by
  [`torch::sampler()`](https://torch.mlverse.org/docs/reference/sampler.html)),
  not an instance, as it is instantiated with the training dataset
  internally.

- `batch_sampler` ::
  [`torch::sampler`](https://torch.mlverse.org/docs/reference/sampler.html)  
  Object that defines how the dataloader draws batches. As for
  `sampler`, this must be the generator. When it is provided, the
  parameters `batch_size`, `shuffle` and `drop_last` are ignored during
  training, because the batch sampler already determines the batches.

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
  Default is `FALSE`. It is ignored when a `batch_sampler` is provided.

- `timeout` :: `numeric(1)`  
  The timeout value for collecting a batch from workers. Negative values
  mean no timeout and the default is `-1`.

- `worker_init_fn` :: `function(id)`  
  A function that receives the worker id (in `[1, num_workers]`) and is
  executed after seeding on the worker but before data loading.

- `worker_globals` :: [`list()`](https://rdrr.io/r/base/list.html) \|
  [`character()`](https://rdrr.io/r/base/character.html)  
  When loading data in parallel, this allows to export globals to the
  workers. If this is a character vector, the objects in the global
  environment with those names are copied to the workers.

- `worker_packages` ::
  [`character()`](https://rdrr.io/r/base/character.html)  
  Which packages to load on the workers.

Also see
[`torch::dataloader`](https://torch.mlverse.org/docs/reference/dataloader.html)
for more information.

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
`$train()` method is called on the
[`Task`](https://mlr3.mlr-org.com/reference/Task.html) stored in the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3pipelines::PipeOpLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_learner.html)
-\> `PipeOpTorchModel`

## Methods

### Public methods

- [`PipeOpTorchModel$new()`](#method-PipeOpTorchModel-initialize)

- [`PipeOpTorchModel$clone()`](#method-PipeOpTorchModel-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchModel$new()`

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

### `PipeOpTorchModel$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchModel$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
