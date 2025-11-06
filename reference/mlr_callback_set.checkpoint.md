# Checkpoint Callback

Saves the optimizer and network states during training. The final
network and optimizer are always stored.

## Details

Saving the learner itself in the callback with a trained model is
impossible, as the model slot is set *after* the last callback step is
executed.

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)

## Super class

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
-\> `CallbackSetCheckpoint`

## Methods

### Public methods

- [`CallbackSetCheckpoint$new()`](#method-CallbackSetCheckpoint-new)

- [`CallbackSetCheckpoint$on_epoch_end()`](#method-CallbackSetCheckpoint-on_epoch_end)

- [`CallbackSetCheckpoint$on_batch_end()`](#method-CallbackSetCheckpoint-on_batch_end)

- [`CallbackSetCheckpoint$on_exit()`](#method-CallbackSetCheckpoint-on_exit)

- [`CallbackSetCheckpoint$clone()`](#method-CallbackSetCheckpoint-clone)

Inherited methods

- [`mlr3torch::CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-load_state_dict)
- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-print)
- [`mlr3torch::CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetCheckpoint$new(path, freq, freq_type = "epoch")

#### Arguments

- `path`:

  (`character(1)`)  
  The path to a folder where the models are saved.

- `freq`:

  (`integer(1)`)  
  The frequency how often the model is saved. Frequency is either per
  step or epoch, which can be configured through the `freq_type`
  parameter.

- `freq_type`:

  (`character(1)`)  
  Can be be either `"epoch"` (default) or `"step"`.

------------------------------------------------------------------------

### Method `on_epoch_end()`

Saves the network and optimizer state dict. Does nothing if `freq_type`
or `freq` are not met.

#### Usage

    CallbackSetCheckpoint$on_epoch_end()

------------------------------------------------------------------------

### Method `on_batch_end()`

Saves the selected objects defined in `save`. Does nothing if freq_type
or freq are not met.

#### Usage

    CallbackSetCheckpoint$on_batch_end()

------------------------------------------------------------------------

### Method `on_exit()`

Saves the learner.

#### Usage

    CallbackSetCheckpoint$on_exit()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetCheckpoint$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
cb = t_clbk("checkpoint", freq = 1)
task = tsk("iris")

pth = tempfile()
learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
learner$param_set$set_values(cb.checkpoint.path = pth)

learner$train(task)

list.files(pth)
#> [1] "network1.pt"   "network2.pt"   "network3.pt"   "optimizer1.pt"
#> [5] "optimizer2.pt" "optimizer3.pt"
```
