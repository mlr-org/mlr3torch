# Create a Set of Callbacks for Torch

Creates an `R6ClassGenerator` inheriting from
[`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md).
Additionally performs checks such as that the stages are not
accidentally misspelled. To create a
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)
use
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md).

In order for the resulting class to be cloneable, the private method
`$deep_clone()` must be provided.

## Usage

``` r
callback_set(
  classname,
  on_begin = NULL,
  on_end = NULL,
  on_exit = NULL,
  on_epoch_begin = NULL,
  on_before_valid = NULL,
  on_epoch_end = NULL,
  on_batch_begin = NULL,
  on_batch_end = NULL,
  on_after_backward = NULL,
  on_batch_valid_begin = NULL,
  on_batch_valid_end = NULL,
  on_valid_end = NULL,
  state_dict = NULL,
  load_state_dict = NULL,
  initialize = NULL,
  public = NULL,
  private = NULL,
  active = NULL,
  parent_env = parent.frame(),
  inherit = CallbackSet,
  lock_objects = FALSE
)
```

## Arguments

- classname:

  (`character(1)`)  
  The class name.

- on_begin, on_end, on_epoch_begin, on_before_valid, on_epoch_end,
  on_batch_begin, on_batch_end, on_after_backward, on_batch_valid_begin,
  on_batch_valid_end, on_valid_end, on_exit:

  (`function`)  
  Function to execute at the given stage, see section *Stages*.

- state_dict:

  (`function()`)  
  The function that retrieves the state dict from the callback. This is
  what will be available in the learner after training.

- load_state_dict:

  (`function(state_dict)`)  
  Function that loads a callback state.

- initialize:

  (`function()`)  
  The initialization method of the callback.

- public, private, active:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Additional public, private, and active fields to add to the callback.

- parent_env:

  ([`environment()`](https://rdrr.io/r/base/environment.html))  
  The parent environment for the
  [`R6Class`](https://r6.r-lib.org/reference/R6Class.html).

- inherit:

  (`R6ClassGenerator`)  
  From which class to inherit. This class must either be
  [`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
  (default) or inherit from it.

- lock_objects:

  (`logical(1)`)  
  Whether to lock the objects of the resulting
  [`R6Class`](https://r6.r-lib.org/reference/R6Class.html). If `FALSE`
  (default), values can be freely assigned to `self` without declaring
  them in the class definition.

## Value

[`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)
