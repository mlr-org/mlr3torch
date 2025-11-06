# Unfreezing Weights Callback

Unfreeze some weights (parameters of the network) after some number of
steps or epochs.

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
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)

## Super class

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
-\> `CallbackSetUnfreeze`

## Methods

### Public methods

- [`CallbackSetUnfreeze$new()`](#method-CallbackSetUnfreeze-new)

- [`CallbackSetUnfreeze$on_begin()`](#method-CallbackSetUnfreeze-on_begin)

- [`CallbackSetUnfreeze$on_epoch_begin()`](#method-CallbackSetUnfreeze-on_epoch_begin)

- [`CallbackSetUnfreeze$on_batch_begin()`](#method-CallbackSetUnfreeze-on_batch_begin)

- [`CallbackSetUnfreeze$clone()`](#method-CallbackSetUnfreeze-clone)

Inherited methods

- [`mlr3torch::CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-load_state_dict)
- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-print)
- [`mlr3torch::CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetUnfreeze$new(starting_weights, unfreeze)

#### Arguments

- `starting_weights`:

  (`Select`)  
  A `Select` denoting the weights that are trainable from the start.

- `unfreeze`:

  (`data.table`)  
  A `data.table` with a column `weights` (a list column of `Select`s)
  and a column `epoch` or `batch`. The selector indicates which
  parameters to unfreeze, while the `epoch` or `batch` column indicates
  when to do so.

------------------------------------------------------------------------

### Method `on_begin()`

Sets the starting weights

#### Usage

    CallbackSetUnfreeze$on_begin()

------------------------------------------------------------------------

### Method `on_epoch_begin()`

Unfreezes weights if the training is at the correct epoch

#### Usage

    CallbackSetUnfreeze$on_epoch_begin()

------------------------------------------------------------------------

### Method `on_batch_begin()`

Unfreezes weights if the training is at the correct batch

#### Usage

    CallbackSetUnfreeze$on_batch_begin()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetUnfreeze$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
task = tsk("iris")
cb = t_clbk("unfreeze")
mlp = lrn("classif.mlp", callbacks = cb,
 cb.unfreeze.starting_weights = select_invert(
   select_name(c("0.weight", "3.weight", "6.weight", "6.bias"))
 ),
 cb.unfreeze.unfreeze = data.table(
   epoch = c(2, 5),
   weights = list(select_name("0.weight"), select_name(c("3.weight", "6.weight")))
 ),
 epochs = 6, batch_size = 150, neurons = c(1, 1, 1)
)

mlp$train(task)
```
