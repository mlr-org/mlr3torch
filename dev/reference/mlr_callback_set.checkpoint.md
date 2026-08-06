# Checkpoint Callback

Saves the optimizer and network states during training. The final
network and optimizer are always stored.

Checkpoints are written at the end of an epoch. For one written after
epoch `<n>`, two files are created in `path`:

- `network<n>.pt` :: The `$state_dict()` of the network.

- `optimizer<n>.pt` :: The `$state_dict()` of the optimizer.

An epoch that was interrupted – because training failed or was stopped –
is not written under its own number, so `network<n>.pt` is always the
network at the *end* of epoch `n`.

## Details

Saving the learner itself in the callback with a trained model is
impossible, as the model slot is set *after* the last callback step is
executed.

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/dev/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)

## Super class

[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\> `CallbackSetCheckpoint`

## Public fields

- `weight`:

  (`numeric(1)`)  
  `Inf`, so that this callback runs after the other callbacks and hence
  saves the network and optimizer as they are at the end of the stage,
  see section *Ordering* of
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md).
  The only exception is the restore of `restore_best_weights`, which
  happens afterwards, so a checkpoint always holds the network as
  training left it.

## Methods

### Public methods

- [`CallbackSetCheckpoint$new()`](#method-CallbackSetCheckpoint-initialize)

- [`CallbackSetCheckpoint$on_epoch_end()`](#method-CallbackSetCheckpoint-on_epoch_end)

- [`CallbackSetCheckpoint$on_exit()`](#method-CallbackSetCheckpoint-on_exit)

- [`CallbackSetCheckpoint$clone()`](#method-CallbackSetCheckpoint-clone)

Inherited methods

- [`CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-load_state_dict)
- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)
- [`CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### `CallbackSetCheckpoint$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetCheckpoint$new(path, freq)

#### Arguments

- `path`:

  (`character(1)`)  
  The path to a folder where the models are saved.

- `freq`:

  (`integer(1)`)  
  How often the model is saved, in epochs.

------------------------------------------------------------------------

### `CallbackSetCheckpoint$on_epoch_end()`

Saves the network and optimizer state dict. Does nothing if `freq` is
not met.

#### Usage

    CallbackSetCheckpoint$on_epoch_end()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$on_exit()`

Saves the final network and optimizer, unless the last complete epoch
was already saved.

#### Usage

    CallbackSetCheckpoint$on_exit()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$clone()`

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
