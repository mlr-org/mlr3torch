# Progress Callback

Prints a progress bar and the metrics for training and validation.

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

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\> `CallbackSetProgress`

## Methods

### Public methods

- [`CallbackSetProgress$new()`](#method-CallbackSetProgress-new)

- [`CallbackSetProgress$on_epoch_begin()`](#method-CallbackSetProgress-on_epoch_begin)

- [`CallbackSetProgress$on_batch_end()`](#method-CallbackSetProgress-on_batch_end)

- [`CallbackSetProgress$on_before_valid()`](#method-CallbackSetProgress-on_before_valid)

- [`CallbackSetProgress$on_batch_valid_end()`](#method-CallbackSetProgress-on_batch_valid_end)

- [`CallbackSetProgress$on_epoch_end()`](#method-CallbackSetProgress-on_epoch_end)

- [`CallbackSetProgress$on_end()`](#method-CallbackSetProgress-on_end)

- [`CallbackSetProgress$clone()`](#method-CallbackSetProgress-clone)

Inherited methods

- [`mlr3torch::CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-load_state_dict)
- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)
- [`mlr3torch::CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetProgress$new(digits = 2)

#### Arguments

- `digits`:

  `integer(1)`  
  The number of digits to print for the measures.

------------------------------------------------------------------------

### Method `on_epoch_begin()`

Initializes the progress bar for training.

#### Usage

    CallbackSetProgress$on_epoch_begin()

------------------------------------------------------------------------

### Method `on_batch_end()`

Increments the training progress bar.

#### Usage

    CallbackSetProgress$on_batch_end()

------------------------------------------------------------------------

### Method `on_before_valid()`

Creates the progress bar for validation.

#### Usage

    CallbackSetProgress$on_before_valid()

------------------------------------------------------------------------

### Method `on_batch_valid_end()`

Increments the validation progress bar.

#### Usage

    CallbackSetProgress$on_batch_valid_end()

------------------------------------------------------------------------

### Method `on_epoch_end()`

Prints a summary of the training and validation process.

#### Usage

    CallbackSetProgress$on_epoch_end()

------------------------------------------------------------------------

### Method `on_end()`

Prints the time at the end of training.

#### Usage

    CallbackSetProgress$on_end()

------------------------------------------------------------------------

### Method `clone()`

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
#> Epoch 1 started (2026-02-08 06:07:40)
#> Validation for epoch 1 started (2026-02-08 06:07:41)
#> 
#> [Summary epoch 1]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.36
#>  * classif.ce = 0.64
#> Measures (Valid):
#>  * classif.ce = 0.73
#> 
#> Epoch 2 started (2026-02-08 06:07:41)
#> Validation for epoch 2 started (2026-02-08 06:07:41)
#> 
#> [Summary epoch 2]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.36
#>  * classif.ce = 0.64
#> Measures (Valid):
#>  * classif.ce = 0.73
#> 
#> Epoch 3 started (2026-02-08 06:07:41)
#> Validation for epoch 3 started (2026-02-08 06:07:42)
#> 
#> [Summary epoch 3]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.36
#>  * classif.ce = 0.64
#> Measures (Valid):
#>  * classif.ce = 0.73
#> 
#> Epoch 4 started (2026-02-08 06:07:42)
#> Validation for epoch 4 started (2026-02-08 06:07:42)
#> 
#> [Summary epoch 4]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.36
#>  * classif.ce = 0.64
#> Measures (Valid):
#>  * classif.ce = 0.73
#> 
#> Epoch 5 started (2026-02-08 06:07:42)
#> Validation for epoch 5 started (2026-02-08 06:07:43)
#> 
#> [Summary epoch 5]
#> ------------------
#> Measures (Train):
#>  * classif.acc = 0.36
#>  * classif.ce = 0.64
#> Measures (Valid):
#>  * classif.ce = 0.73
#> 
#> Finished training for 5 epochs (2026-02-08 06:07:43)
```
