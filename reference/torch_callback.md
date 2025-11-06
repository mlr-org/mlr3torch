# Create a Callback Descriptor

Convenience function to create a custom
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md).
All arguments that are available in
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md)
are also available here. For more information on how to correctly
implement a new callback, see
[`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md).

## Usage

``` r
torch_callback(
  id,
  classname = paste0("CallbackSet", capitalize(id)),
  param_set = NULL,
  packages = NULL,
  label = capitalize(id),
  man = NULL,
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

- id:

  (`character(1)`)  
  \`  
  The id for the torch callback.

- classname:

  (`character(1)`)  
  The class name.

- param_set:

  (`ParamSet`)  
  The parameter set, if not present it is inferred from the
  `$initialize()` method.

- packages:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  `The packages the callback depends on. Default is`NULL\`.

- label:

  (`character(1)`)  
  The label for the torch callback. Defaults to the capitalized `id`.

- man:

  (`character(1)`)  
  String in the format `[pkg]::[topic]` pointing to a manual page for
  this object. The referenced help package can be opened via method
  `$help()`. The default is `NULL`.

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

[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)

## Internals

It first creates an `R6` class inheriting from
[`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
(using
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md))
and then wraps this generator in a
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)
that can be passed to a torch learner.

## Stages

- `begin` :: Run before the training loop begins.

- `epoch_begin` :: Run he beginning of each epoch.

- `batch_begin` :: Run before the forward call.

- `after_backward` :: Run after the backward call.

- `batch_end` :: Run after the optimizer step.

- `batch_valid_begin` :: Run before the forward call in the validation
  loop.

- `batch_valid_end` :: Run after the forward call in the validation
  loop.

- `valid_end` :: Run at the end of validation.

- `epoch_end` :: Run at the end of each epoch.

- `end` :: Run after last epoch.

- `exit` :: Run at last, using
  [`on.exit()`](https://rdrr.io/r/base/on.exit.html).

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
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md)

## Examples

``` r
custom_tcb = torch_callback("custom",
  initialize = function(name) {
    self$name = name
  },
  on_begin = function() {
    cat("Hello", self$name, ", we will train for ", self$ctx$total_epochs, "epochs.\n")
  },
  on_end = function() {
    cat("Training is done.")
  }
)

learner = lrn("classif.torch_featureless",
  batch_size = 16,
  epochs = 1,
  callbacks = custom_tcb,
  cb.custom.name = "Marie",
  device = "cpu"
)
task = tsk("iris")
learner$train(task)
#> Hello Marie , we will train for  1 epochs.
#> Training is done.
```
