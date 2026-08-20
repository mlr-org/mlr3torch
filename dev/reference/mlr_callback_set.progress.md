# Progress Callback

Prints a progress bar and the metrics for training and validation.

## Resuming

A resumed run prints only the epochs it trains itself, numbered as what
they are: it starts at the epoch after the checkpoint, not at epoch 1.
The time training has taken is carried across runs, so the total this
reports when training ends covers the runs the checkpoint came from as
well and not only the last one. Such a run reports that total split into
the time before it and the time it took itself.

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/dev/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)

## Super class

[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\> `CallbackSetProgress`

## Methods

### Public methods

- [`CallbackSetProgress$new()`](#method-CallbackSetProgress-initialize)

- [`CallbackSetProgress$on_begin()`](#method-CallbackSetProgress-on_begin)

- [`CallbackSetProgress$on_epoch_begin()`](#method-CallbackSetProgress-on_epoch_begin)

- [`CallbackSetProgress$on_batch_end()`](#method-CallbackSetProgress-on_batch_end)

- [`CallbackSetProgress$on_before_valid()`](#method-CallbackSetProgress-on_before_valid)

- [`CallbackSetProgress$on_batch_valid_end()`](#method-CallbackSetProgress-on_batch_valid_end)

- [`CallbackSetProgress$on_epoch_end()`](#method-CallbackSetProgress-on_epoch_end)

- [`CallbackSetProgress$on_end()`](#method-CallbackSetProgress-on_end)

- [`CallbackSetProgress$state_dict()`](#method-CallbackSetProgress-state_dict)

- [`CallbackSetProgress$load_state_dict()`](#method-CallbackSetProgress-load_state_dict)

- [`CallbackSetProgress$clone()`](#method-CallbackSetProgress-clone)

Inherited methods

- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)

------------------------------------------------------------------------

### `CallbackSetProgress$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetProgress$new(digits = 2)

#### Arguments

- `digits`:

  `integer(1)`  
  The number of digits to print for the measures.

------------------------------------------------------------------------

### `CallbackSetProgress$on_begin()`

Starts this run's timer.

#### Usage

    CallbackSetProgress$on_begin()

------------------------------------------------------------------------

### `CallbackSetProgress$on_epoch_begin()`

Initializes the progress bar for training.

#### Usage

    CallbackSetProgress$on_epoch_begin()

------------------------------------------------------------------------

### `CallbackSetProgress$on_batch_end()`

Increments the training progress bar.

#### Usage

    CallbackSetProgress$on_batch_end()

------------------------------------------------------------------------

### `CallbackSetProgress$on_before_valid()`

Creates the progress bar for validation.

#### Usage

    CallbackSetProgress$on_before_valid()

------------------------------------------------------------------------

### `CallbackSetProgress$on_batch_valid_end()`

Increments the validation progress bar.

#### Usage

    CallbackSetProgress$on_batch_valid_end()

------------------------------------------------------------------------

### `CallbackSetProgress$on_epoch_end()`

Prints a summary of the training and validation process.

#### Usage

    CallbackSetProgress$on_epoch_end()

------------------------------------------------------------------------

### `CallbackSetProgress$on_end()`

Prints the time at the end of training, and how long training took in
total. A resumed run also reports how much of that total it contributed
itself.

#### Usage

    CallbackSetProgress$on_end()

------------------------------------------------------------------------

### `CallbackSetProgress$state_dict()`

Returns the seconds trained so far, so that a resumed run reports the
time of all runs together rather than only its own.

#### Usage

    CallbackSetProgress$state_dict()

------------------------------------------------------------------------

### `CallbackSetProgress$load_state_dict()`

Loads the time that the previous runs took.

#### Usage

    CallbackSetProgress$load_state_dict(state_dict)

#### Arguments

- `state_dict`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  The state dict as retrieved via `$state_dict()`.

------------------------------------------------------------------------

### `CallbackSetProgress$clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetProgress$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
task = tsk("iris")

learner = lrn("classif.mlp", epochs = 5, batch_size = 1,
  callbacks = t_clbk("progress"), validate = 0.3)
learner$param_set$set_values(
  measures_train = msrs(c("classif.acc", "classif.ce")),
  measures_valid = msr("classif.ce")
)

learner$train(task)
#> Epoch 1/5 started (2026-08-20 12:26:22)
#> Validation for epoch 1 started (2026-08-20 12:26:22)
#> 
#> [Summary epoch 1]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.29
#>  * classif.ce = 0.71
#> Measures (Valid):
#>  * classif.ce = 0.56
#> 
#> Epoch 2/5 started (2026-08-20 12:26:22)
#> Validation for epoch 2 started (2026-08-20 12:26:22)
#> 
#> [Summary epoch 2]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.33
#>  * classif.ce = 0.67
#> Measures (Valid):
#>  * classif.ce = 0.47
#> 
#> Epoch 3/5 started (2026-08-20 12:26:23)
#> Validation for epoch 3 started (2026-08-20 12:26:23)
#> 
#> [Summary epoch 3]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.64
#>  * classif.ce = 0.36
#> Measures (Valid):
#>  * classif.ce = 0.36
#> 
#> Epoch 4/5 started (2026-08-20 12:26:23)
#> Validation for epoch 4 started (2026-08-20 12:26:23)
#> 
#> [Summary epoch 4]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.68
#>  * classif.ce = 0.32
#> Measures (Valid):
#>  * classif.ce = 0.36
#> 
#> Epoch 5/5 started (2026-08-20 12:26:23)
#> Validation for epoch 5 started (2026-08-20 12:26:24)
#> 
#> [Summary epoch 5]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.68
#>  * classif.ce = 0.32
#> Measures (Valid):
#>  * classif.ce = 0.36
#> 
#> Finished training for 5 epochs (2026-08-20 12:26:24, 2.2s total)
```
