# Base Class for Torch Learners

This base class provides the basic functionality for training and
prediction of a neural network. All torch learners should inherit from
this class.

## Validation

To specify the validation data, you can set the `$validate` field of the
Learner, which can be set to:

- `NULL`: no validation

- `ratio`: only proportion `1 - ratio` of the task is used for training
  and `ratio` is used for validation.

- `"test"` means that the `"test"` task of a resampling is used and is
  not possible when calling `$train()` manually.

- `"predefined"`: This will use the predefined `$internal_valid_task` of
  a [`mlr3::Task`](https://mlr3.mlr-org.com/reference/Task.html).

This validation data can also be used for early stopping, see the
description of the `Learner`'s parameters.

## Saving a Learner

In order to save a `LearnerTorch` for later usage, it is necessary to
call the `$marshal()` method on the `Learner` before writing it to disk,
as the object will otherwise not be saved correctly. After loading a
marshaled `LearnerTorch` into R again, you then need to call
`$unmarshal()` to transform it into a useable state.

## Early Stopping and Internal Tuning

In order to prevent overfitting, the `LearnerTorch` class allows to use
early stopping via the `patience` and `min_delta` parameters, see the
`Learner`'s parameters. When tuning a `LearnerTorch` it is also possible
to combine the explicit tuning via `mlr3tuning` and the `LearnerTorch`'s
internal tuning of the epochs via early stopping. To do so, you just
need to include `epochs = to_tune(upper = <upper>, internal = TRUE)` in
the search space, where `<upper>` is the maximally allowed number of
epochs, and configure the early stopping.

## Checkpointing and Resuming

It is possible to save intermediate results from a run via the
[`t_clbk("checkpoint")`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md)
callback. It is then possible to train for more epochs by setting the
`resume` parameter of the `LearnerTorch`. This parameter can either be a
path or `TRUE` which will use the path of the provided checkpoint
callback. Only the number of epochs should be changed between resumed
runs, other parameter changes are considered undefined behavior. Also,
make sure to use the same train-validation split. When the latest
written checkpoint was for `n1` epochs, the learner needs to be
configured to be trained for `n >= n1` epochs and the training will run
for `n2 = n - n1` epochs. With `n = n1` the checkpointed run is already
finished, so nothing is trained and the model of the checkpoint is
returned – which is what lets a script that restarts itself be run again
after it succeeded, and what recovers the model of a run that was killed
after its last epoch. Configuring `n < n1` is an error. Resuming will
load the network weights, optimizer states and callback states. For some
callbacks, training for `n1` and then `n2` epochs via resuming is not
the same as training for `n` epochs from the start. This is for example
the case for learning rate schedulers that depend on the total number of
epochs to train for. The callbacks document their behavior under a
corresponding *Resuming* section in their documentation. Furthermore,
rng states are not restored, which constitutes another difference
between a full and a resumed training run.

## Network Head and Target Encoding

Torch learners are expected to have the following output:

- binary classification: `(batch_size, 1)`, representing the logits for
  the positive class.

- multiclass classification: `(batch_size, n_classes)`, representing the
  logits for all classes.

- regression: `(batch_size, 1)` representing the response prediction.

A network may return more than one tensor, in which case it returns a
[`list()`](https://rdrr.io/r/base/list.html) of them. There are two
typical reasons for this:

- Networks with auxiliary classifiers such as
  [`Inception v3`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torchvision.md)
  return additional predictions that only exist to contribute to the
  loss during training. Here, every element has the shape given above
  and the first one is the prediction of interest.

- A prediction that consists of several quantities – e.g. a mean and a
  standard deviation – is expressed by returning one tensor per
  quantity, which the prediction encoding combines.

In both cases the complete output is what the rest of the learner works
with:

- The loss is applied to it. Because the configured loss expects a
  single tensor, a learner whose network returns a list has to wrap it
  by overloading `.loss_fn()`, see the list of methods below.
  [`ContextTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_context_torch.md)
  makes the output available as `ctx$y_hats`.

- The prediction is encoded from it, both when predicting and when
  calculating the training and validation scores, so
  `.encode_prediction()` always receives the complete network output.
  Such a learner needs an
  [`encode_prediction()`](https://mlr3torch.mlr-org.com/dev/reference/encode_prediction.md)
  method for its task type, or has to overload the private
  `.encode_prediction()` method, because the encodings of the built-in
  task types expect a single tensor.

Note that the complete output is whatever the network returned in the
mode it was called in, so a network whose extra tensors exist only
during training – as auxiliary classifiers do – returns a different
structure during training than during prediction, and
`.encode_prediction()` has to handle both. `classif.inception_v3` does
this by encoding only the prediction of the main classifier.

Furthermore, the target encoding is expected to be as follows:

- regression: The `numeric` target variable of a
  [`TaskRegr`](https://mlr3.mlr-org.com/reference/TaskRegr.html) is
  encoded as a
  [`torch_float`](https://torch.mlverse.org/docs/reference/torch_dtype.html)
  with shape `c(batch_size, 1)`.

- binary classification: The `factor` target variable of a
  [`TaskClassif`](https://mlr3.mlr-org.com/reference/TaskClassif.html)
  is encoded as a
  [`torch_float`](https://torch.mlverse.org/docs/reference/torch_dtype.html)
  with shape `(batch_size, 1)` where the positive class
  (`Task$positive`, which is also ensured to be the first factor level)
  is `1` and the negative class is `0`.

- multi-class classification: The `factor` target variable of a
  [`TaskClassif`](https://mlr3.mlr-org.com/reference/TaskClassif.html)
  is a label-encoded
  [`torch_long`](https://torch.mlverse.org/docs/reference/torch_dtype.html)
  with shape `(batch_size)` where the label-encoding goes from `1` to
  `n_classes`.

## Predicting Tensors

The predict type `"lazy_tensor"`, available for the task type `"torch"`,
hands back what the network produced – a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
with one element per observation – instead of asking the task's
`default_encoder` to turn it into a response. It is how to get at the
logits of a classifier or the reconstruction of an autoencoder, and a
task predicted this way needs no encoder at all. Unlike `"prob"` and
`"se"` it is not opt-in: every learner for this task type has it among
its `$predict_types`, because handing the output back does not depend on
how the learner encodes a prediction. Set
`learner$predict_type = "lazy_tensor"` to use it. A network with more
than one head hands back one
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
per head, held in a
[`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html) with
one column per head so that the prediction is still one row per
observation;
[`as.data.table()`](https://rdrr.io/pkg/data.table/man/as.data.table.html)
spreads it into `lazy_tensor.<head>` columns.

Two things to know before using it:

- **Such a prediction does not survive
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html).** It holds `torch`
  tensors, which are external pointers: saving *succeeds*, and the
  object then fails with *external pointer is not valid* the next time
  the tensors are touched, in this session or in another. This applies
  to a
  [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)
  holding one as well – its row ids and scores survive, its tensors do
  not.

- Nothing about it is lazy. A
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  built from a tensor holds that tensor, so a prediction of this type is
  the network's output in memory, and
  [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) holds
  every fold's – combining the folds concatenates them, since lazy
  tensors from different networks share no data descriptor and cannot be
  concatenated lazily.

## Important Runtime Considerations

There are a few hyperparameters settings that can have a considerable
impact on the runtime of the learner. These include:

- `device`: Use a GPU if possible.

- `num_threads`: Set this to the number of CPU cores available if
  training on CPU. When resampling, benchmarking or tuning in parallel,
  each worker uses `num_threads` threads, so divide the available cores
  among the workers instead to avoid oversubscribing the machine.

- `tensor_dataset`: Set this to `TRUE` (or `"device"` if on a GPU) if
  the dataset fits into memory.

- `batch_size`: Especially for very small models, choose a larger batch
  size.

Also, see the *Early Stopping and Internal Tuning* section for how to
terminate training early.

## Model

The Model is a list of class `"learner_torch_model"` with the following
elements:

- `network` :: The trained
  [network](https://torch.mlverse.org/docs/reference/nn_module.html).

- `optimizer` :: The `$state_dict()`
  [optimizer](https://torch.mlverse.org/docs/reference/optimizer.html)
  used to train the network.

- `loss_fn` :: The `$state_dict()` of the
  [loss](https://torch.mlverse.org/docs/reference/nn_module.html) used
  to train the network.

- `callbacks` :: The
  [callbacks](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
  used to train the network.

- `seed` :: The seed that was / is used for training and prediction.

- `epochs` :: How many epochs the model was trained for (early
  stopping).

- `task_col_info` :: A
  [`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html)
  containing information about the train-task.

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

## Inheriting

There are no separate classes for classification and regression to
inherit from. Instead, the `task_type` must be specified as a
construction argument. Any task type that is registered in
[`mlr_reflections$task_types`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
can be used. Support for a task type that
[mlr3torch](https://CRAN.R-project.org/package=mlr3torch) does not know
is added by implementing methods for the three S3 generics that hold the
task-type-specific behaviour:
[`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md)
(how many output neurons the network needs),
[`get_target_batchgetter()`](https://mlr3torch.mlr-org.com/dev/reference/get_target_batchgetter.md)
(how the target is turned into a tensor) and
[`encode_prediction()`](https://mlr3torch.mlr-org.com/dev/reference/encode_prediction.md)
(how the network's output is turned back into a prediction). Such a
learner also has to be given a `loss` explicitly. This class can also be
used for custom task types, see
[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md)
and the *Custom Learning Problems* article for more information.

When inheriting from this class, one should overload the following
methods:

- `.network(task, param_vals)`  
  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html),
  [`list()`](https://rdrr.io/r/base/list.html)) -\>
  [`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)  
  Construct a
  [`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  object for the given task and parameter values, i.e. the neural
  network that is trained by the learner. Note that a specific output
  shape is expected from the returned network, see section *Network Head
  and Target Encoding*. That section also describes when a network can
  return more than one tensor. You can use
  [`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md)
  to obtain the correct output dimension for a given task.

- `.loss_fn(task, param_vals)`  
  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html),
  [`list()`](https://rdrr.io/r/base/list.html)) -\>
  [`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)  
  Construct the loss that is applied to the output of the network. The
  default implementation generates the loss that was configured by the
  user, i.e. `self$loss$generate(task)`. Overload this if the network
  returns more than one prediction and the configured loss has to be
  wrapped, see the `aux_logits` parameter of
  [`classif.inception_v3`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torchvision.md).

- `.ingress_tokens(task, param_vals)`  
  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html),
  [`list()`](https://rdrr.io/r/base/list.html)) -\> named
  [`list()`](https://rdrr.io/r/base/list.html) with
  [`TorchIngressToken`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md)s  
  Create the
  [`TorchIngressToken`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md)s
  that are passed to the
  [`task_dataset`](https://mlr3torch.mlr-org.com/dev/reference/task_dataset.md)
  constructor. The number of ingress tokens must correspond to the
  number of input parameters of the network. If there is more than one
  input, the names must correspond to the inputs of the network. See
  [`ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/ingress_num.md),
  [`ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/ingress_categ.md),
  and
  [`ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/ingress_ltnsr.md)
  on how to easily create the correct tokens. For more flexibility, you
  can also directly implement the `.dataset(task, param_vals)` method,
  see below.

- `.dataset(task, param_vals)`  
  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html),
  [`list()`](https://rdrr.io/r/base/list.html)) -\>
  [`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html)  
  Create the dataset for the task. Don't implement this if the
  `.ingress_tokens()` method is defined. The dataset must return a named
  list where:

  - `x` is a list of torch tensors that are the input to the network.
    For networks with more than one input, the names must correspond to
    the inputs of the network.

  - `y` is the target tensor.

  - `.index` are the indices of the batch
    ([`integer()`](https://rdrr.io/r/base/integer.html) or a
    [`torch_int()`](https://torch.mlverse.org/docs/reference/torch_dtype.html)).

  For information on the expected target encoding of `y`, see section
  *Network Head and Target Encoding*. Moreover, one needs to pay
  attention respect the row ids of the provided task. It is recommended
  to relu on
  [`task_dataset`](https://mlr3torch.mlr-org.com/dev/reference/task_dataset.md)
  for creating the
  [`dataset`](https://torch.mlverse.org/docs/reference/dataset.html).

It is also possible to overwrite the private `.dataloader()` method.
This must respect the dataloader parameters from the
[`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html).

- `.dataloader(dataset, param_vals)`  
  ([`dataset`](https://torch.mlverse.org/docs/reference/dataset.html),
  [`list()`](https://rdrr.io/r/base/list.html)) -\>
  [`torch::dataloader`](https://torch.mlverse.org/docs/reference/dataloader.html)  
  Create a dataloader from the dataset. Needs to respect at least
  `batch_size` and `shuffle` (otherwise predictions will be incorrectly
  ordered). Use `get_batch_size(param_vals, "train")` to obtain the
  batch size for the respective phase, which takes the
  `batch_size_predict` parameter into account.

To change the predict types, it is possible to overwrite the method
below:

- `.encode_prediction(network_output, task)`  
  ([`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  or [`list()`](https://rdrr.io/r/base/list.html) of them,
  [`Task`](https://mlr3.mlr-org.com/reference/Task.html)) -\>
  [`list()`](https://rdrr.io/r/base/list.html)  
  Take in the raw predictions from `self$network` (`network_output`) and
  encode them into a format that can be converted to valid `mlr3`
  predictions using
  [`mlr3::as_prediction_data()`](https://mlr3.mlr-org.com/reference/as_prediction_data.html).
  It is a [`list()`](https://rdrr.io/r/base/list.html) of tensors when
  the network returns more than one, see section *Network Head and
  Target Encoding*. This method must take `self$predict_type` into
  account.

While it is possible to add parameters by specifying the `param_set`
construction argument, it is currently not possible to remove existing
parameters, i.e. those listed in section *Parameters*. None of the
parameters provided in `param_set` can have an id that starts with
`"loss."`, `"opt."`, or `"cb."`, as these are preserved for the
dynamically constructed parameters of the optimizer, the loss function,
and the callbacks.

To perform additional input checks on the task, the private
`.check_train_task(task, param_vals)` and
`.check_predict_task(task, param_vals)` can be overwritten. These should
return `TRUE` if the input task is valid and otherwise a string with an
error message.

For learners that have other construction arguments that should change
the hash of a learner, it is required to implement the private
`$.additional_phash_input()`.

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.tabm`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tabm.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)

## Super class

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
`LearnerTorch`

## Active bindings

- `validate`:

  How to construct the internal validation data. This parameter can be
  either `NULL`, a ratio in \$(0, 1)\$, `"test"`, or `"predefined"`.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md))  
  The torch loss.

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md))  
  The torch optimizer.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md)s)  
  List of torch callbacks. The ids will be set as the names.

- `internal_valid_scores`:

  Retrieves the internal validation scores as a named
  [`list()`](https://rdrr.io/r/base/list.html). Specify the `$validate`
  field and the `measures_valid` parameter to configure this. Returns
  `NULL` if learner is not trained yet.

- `internal_tuned_values`:

  When early stopping is active, this returns a named list with the
  early-stopped epochs, otherwise an empty list is returned. Returns
  `NULL` if learner is not trained yet.

- `marshaled`:

  (`logical(1)`)  
  Whether the learner is marshaled.

- `network`:

  ([`nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  Shortcut for `learner$model$network`.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  The parameter set

- `hash`:

  (`character(1)`)  
  Hash (unique identifier) for this object.

- `phash`:

  (`character(1)`)  
  Hash (unique identifier) for this partial object, excluding some
  components which are varied systematically during tuning (parameter
  values).

## Methods

### Public methods

- [`LearnerTorch$new()`](#method-LearnerTorch-initialize)

- [`LearnerTorch$format()`](#method-LearnerTorch-format)

- [`LearnerTorch$print()`](#method-LearnerTorch-print)

- [`LearnerTorch$marshal()`](#method-LearnerTorch-marshal)

- [`LearnerTorch$unmarshal()`](#method-LearnerTorch-unmarshal)

- [`LearnerTorch$dataset()`](#method-LearnerTorch-dataset)

- [`LearnerTorch$clone()`](#method-LearnerTorch-clone)

Inherited methods

- [`mlr3::Learner$base_learner()`](https://mlr3.mlr-org.com/reference/Learner.html#method-base_learner)
- [`mlr3::Learner$configure()`](https://mlr3.mlr-org.com/reference/Learner.html#method-configure)
- [`mlr3::Learner$encapsulate()`](https://mlr3.mlr-org.com/reference/Learner.html#method-encapsulate)
- [`mlr3::Learner$help()`](https://mlr3.mlr-org.com/reference/Learner.html#method-help)
- [`mlr3::Learner$predict()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict)
- [`mlr3::Learner$predict_newdata()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict_newdata)
- [`mlr3::Learner$reset()`](https://mlr3.mlr-org.com/reference/Learner.html#method-reset)
- [`mlr3::Learner$selected_features()`](https://mlr3.mlr-org.com/reference/Learner.html#method-selected_features)
- [`mlr3::Learner$train()`](https://mlr3.mlr-org.com/reference/Learner.html#method-train)

------------------------------------------------------------------------

### `LearnerTorch$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerTorch$new(
      id,
      task_type,
      param_set,
      properties = character(),
      man,
      label,
      feature_types,
      optimizer = NULL,
      loss = NULL,
      packages = character(),
      predict_types = NULL,
      callbacks = list(),
      jittable = FALSE
    )

#### Arguments

- `id`:

  (`character(1)`)  
  The id for of the new object.

- `task_type`:

  (`character(1)`)  
  The task type.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) or
  [`alist()`](https://rdrr.io/r/base/list.html))  
  Either a parameter set, or an
  [`alist()`](https://rdrr.io/r/base/list.html) containing different
  values of self, e.g.
  `alist(private$.param_set1, private$.param_set2)`, from which a
  [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html)
  collection should be created.

- `properties`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The properties of the object. See
  [`mlr_reflections$learner_properties`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values.

- `man`:

  (`character(1)`)  
  String in the format `[pkg]::[topic]` pointing to a manual page for
  this object. The referenced help package can be opened via method
  `$help()`.

- `label`:

  (`character(1)`)  
  Label for the new instance.

- `feature_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The feature types. See
  [`mlr_reflections$task_feature_types`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values, Additionally, `"lazy_tensor"` is supported.

- `optimizer`:

  (`NULL` or
  [`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md))  
  The optimizer to use for training. Defaults to adam.

- `loss`:

  (`NULL` or
  [`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md))  
  The loss to use for training. Defaults to MSE for regression and cross
  entropy for classification. For other task types there is no default
  and the loss has to be given, because which loss is appropriate
  depends on the learning problem.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `predict_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The predict types. See
  [`mlr_reflections$learner_predict_types`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values. For regression, the default is `"response"`. For
  classification, this defaults to `"response"` and `"prob"`. For the
  task type `"torch"`, it defaults to `"response"`. For other task
  types, it defaults to all predict types that are registered for the
  task type. To deviate from the defaults, it is necessary to overwrite
  the private `$.encode_prediction()` method, see section *Inheriting*.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md)s)  
  The callbacks to use for training. Defaults to an
  empty` `[`list()`](https://rdrr.io/r/base/list.html), i.e. no
  callbacks. Within a stage they are called in the order in which they
  are provided, unless a callback requests otherwise via its `$weight`,
  see section *Ordering* of
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md).

- `jittable`:

  (`logical(1)`)  
  Whether the model can be jit-traced. Default is `FALSE`.

------------------------------------------------------------------------

### `LearnerTorch$format()`

Helper for print outputs.

#### Usage

    LearnerTorch$format(...)

#### Arguments

- `...`:

  (ignored).

------------------------------------------------------------------------

### `LearnerTorch$print()`

Prints the object.

#### Usage

    LearnerTorch$print(...)

#### Arguments

- `...`:

  (any)  
  Currently unused.

------------------------------------------------------------------------

### `LearnerTorch$marshal()`

Marshal the learner.

#### Usage

    LearnerTorch$marshal(...)

#### Arguments

- `...`:

  (any)  
  Additional parameters.

#### Returns

self

------------------------------------------------------------------------

### `LearnerTorch$unmarshal()`

Unmarshal the learner.

#### Usage

    LearnerTorch$unmarshal(...)

#### Arguments

- `...`:

  (any)  
  Additional parameters.

#### Returns

self

------------------------------------------------------------------------

### `LearnerTorch$dataset()`

Create the dataset for a task.

#### Usage

    LearnerTorch$dataset(task)

#### Arguments

- `task`:

  [`Task`](https://mlr3.mlr-org.com/reference/Task.html)  
  The task

#### Returns

[`dataset`](https://torch.mlverse.org/docs/reference/dataset.html)

------------------------------------------------------------------------

### `LearnerTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
