# Context for Torch Learner

Context for training a torch learner. This is the - mostly read-only -
information callbacks have access to through the argument `ctx`. For
more information on callbacks, see
[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md).

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/dev/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.unfreeze.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)

## Public fields

- `learner`:

  ([`Learner`](https://mlr3.mlr-org.com/reference/Learner.html))  
  The torch learner.

- `task_train`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  The training task.

- `task_valid`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`)  
  The validation task.

- `loader_train`:

  ([`torch::dataloader`](https://torch.mlverse.org/docs/reference/dataloader.html))  
  The data loader for training.

- `loader_valid`:

  ([`torch::dataloader`](https://torch.mlverse.org/docs/reference/dataloader.html))  
  The data loader for validation.

- `measures_train`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html)s)  
  Measures used for training.

- `measures_valid`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html)s)  
  Measures used for validation.

- `network`:

  ([`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The torch network.

- `optimizer`:

  ([`torch::optimizer`](https://torch.mlverse.org/docs/reference/optimizer.html))  
  The optimizer.

- `loss_fn`:

  ([`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The loss function.

- `total_epochs`:

  (`integer(1)`)  
  The total number of epochs the learner is trained for.

- `last_scores_train`:

  (named [`list()`](https://rdrr.io/r/base/list.html) or `NULL`)  
  The scores from the last evaluated epoch, calculated over all rows
  that were trained on in that epoch. Names are the ids of the training
  measures. If
  [`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
  sets `eval_freq` different from `1`, this is `NULL` in all epochs that
  don't evaluate the model.

- `last_scores_valid`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  The scores from the last evaluated epoch, calculated over the complete
  validation task. Names are the ids of the validation measures. If
  [`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
  sets `eval_freq` different from `1`, this is `NULL` in all epochs that
  don't evaluate the model.

- `last_loss`:

  (`numeric(1)`)  
  The loss from the last trainings batch.

- `y_hat`:

  ([`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html))  
  The network's output for the current batch, or its first element if
  the network returns more than one tensor. Provided for the most common
  case where a network returns a single tensor. For the full output, see
  `y_hats`.

- `y_hats`:

  ([`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  or [`list()`](https://rdrr.io/r/base/list.html))  
  The complete output of the network for the current batch, i.e. what
  the loss is applied to. This is a
  [`list()`](https://rdrr.io/r/base/list.html) if the network returns
  more than one tensor and identical to `y_hat` otherwise.

- `epoch`:

  (`integer(1)`)  
  The current epoch.

- `step`:

  (`integer(1)`)  
  The current iteration, i.e. the index of the batch within the current
  epoch. This is reset to `0` at the beginning of every epoch, so a
  globally unique step is `(epoch - 1) * length(loader_train) + step`.

- `prediction_encoder`:

  (`function()`)  
  The learner's prediction encoder.

- `batch`:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  `torch_tensor`s)  
  The current batch.

- `terminate`:

  (`logical(1)`)  
  If this field is set to `TRUE` at the end of an epoch, training stops.

- `callbacks`:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)s)  
  The callbacks that are active during training, named by their ids.
  This allows a callback to access the state of the other callbacks,
  which is for example what
  [`CallbackSetCheckpoint`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md)
  does to save them.

- `device`:

  ([`torch::torch_device`](https://torch.mlverse.org/docs/reference/torch_device.html))  
  The device.

## Methods

### Public methods

- [`ContextTorch$new()`](#method-ContextTorch-initialize)

- [`ContextTorch$clone()`](#method-ContextTorch-clone)

------------------------------------------------------------------------

### `ContextTorch$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    ContextTorch$new(
      learner,
      task_train,
      task_valid = NULL,
      loader_train,
      loader_valid = NULL,
      measures_train = NULL,
      measures_valid = NULL,
      network,
      optimizer,
      loss_fn,
      total_epochs,
      prediction_encoder,
      eval_freq = 1L,
      device
    )

#### Arguments

- `learner`:

  ([`Learner`](https://mlr3.mlr-org.com/reference/Learner.html))  
  The torch learner.

- `task_train`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  The training task.

- `task_valid`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`)  
  The validation task.

- `loader_train`:

  ([`torch::dataloader`](https://torch.mlverse.org/docs/reference/dataloader.html))  
  The data loader for training.

- `loader_valid`:

  ([`torch::dataloader`](https://torch.mlverse.org/docs/reference/dataloader.html)
  or `NULL`)  
  The data loader for validation.

- `measures_train`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html)s or
  `NULL`)  
  Measures used for training. Default is `NULL`.

- `measures_valid`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html)s or
  `NULL`)  
  Measures used for validation.

- `network`:

  ([`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The torch network.

- `optimizer`:

  ([`torch::optimizer`](https://torch.mlverse.org/docs/reference/optimizer.html))  
  The optimizer.

- `loss_fn`:

  ([`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The loss function.

- `total_epochs`:

  (`integer(1)`)  
  The total number of epochs the learner is trained for.

- `prediction_encoder`:

  (`function()`)  
  The learner's prediction encoder. See section *Inheriting* of
  [`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).

- `eval_freq`:

  (`integer(1)`)  
  The evaluation frequency.

- `device`:

  (`character(1)`)  
  The device.

------------------------------------------------------------------------

### `ContextTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    ContextTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
