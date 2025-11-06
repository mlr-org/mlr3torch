# TensorBoard Logging Callback

Logs training loss, training measures, and validation measures as
events. To view them, use TensorBoard with `tensorflow::tensorboard()`
(requires `tensorflow`) or the CLI.

## Details

Logs events at most every epoch.

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)

## Super class

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
-\> `CallbackSetTB`

## Methods

### Public methods

- [`CallbackSetTB$new()`](#method-CallbackSetTB-new)

- [`CallbackSetTB$on_epoch_end()`](#method-CallbackSetTB-on_epoch_end)

- [`CallbackSetTB$clone()`](#method-CallbackSetTB-clone)

Inherited methods

- [`mlr3torch::CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-load_state_dict)
- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-print)
- [`mlr3torch::CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetTB$new(path, log_train_loss)

#### Arguments

- `path`:

  (`character(1)`)  
  The path to a folder where the events are logged. Point TensorBoard to
  this folder to view them.

- `log_train_loss`:

  (`logical(1)`)  
  Whether we log the training loss.

------------------------------------------------------------------------

### Method `on_epoch_end()`

Logs the training loss, training measures, and validation measures as
TensorBoard events.

#### Usage

    CallbackSetTB$on_epoch_end()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetTB$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
