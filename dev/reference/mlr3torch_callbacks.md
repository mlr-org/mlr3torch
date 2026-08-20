# Dictionary of Torch Callbacks

A
[`mlr3misc::Dictionary`](https://mlr3misc.mlr-org.com/reference/Dictionary.html)
of torch callbacks. Use
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md) to
conveniently retrieve callbacks. Can be converted to a
[`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html) using
[`as.data.table`](https://rdrr.io/pkg/data.table/man/as.data.table.html).

## Usage

``` r
mlr3torch_callbacks
```

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/dev/reference/callback_set.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)

Other Dictionary:
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/dev/reference/t_opt.md)

## Examples

``` r
mlr3torch_callbacks$get("checkpoint")
#> <TorchCallback:checkpoint> Checkpoint
#> * Generator: CallbackSetCheckpoint
#> * Parameters: list()
#> * Packages: mlr3torch,torch
# is the same as
t_clbk("checkpoint")
#> <TorchCallback:checkpoint> Checkpoint
#> * Generator: CallbackSetCheckpoint
#> * Parameters: list()
#> * Packages: mlr3torch,torch
# convert to a data.table
as.data.table(mlr3torch_callbacks)
#> Key: <key>
#>                      key                                   label weight
#>                   <char>                                  <char>  <num>
#>  1:           checkpoint                              Checkpoint    Inf
#>  2:              history                                 History      0
#>  3:  lr_cosine_annealing           Cosine Annealing LR Scheduler      0
#>  4:            lr_lambda Multiplication by Function LR Scheduler      0
#>  5:    lr_multiplicative   Multiplication by Factor LR Scheduler      0
#>  6:         lr_one_cycle                     1cycle LR Scheduler      0
#>  7: lr_reduce_on_plateau          Reduce on Plateau LR Scheduler      0
#>  8:              lr_step                 Step Decay LR Scheduler      0
#>  9:             progress                                Progress      0
#> 10:                   tb                             TensorBoard      0
#> 11:             unfreeze                                Unfreeze      0
#>                     packages
#>                       <list>
#>  1:          mlr3torch,torch
#>  2:          mlr3torch,torch
#>  3:          mlr3torch,torch
#>  4:          mlr3torch,torch
#>  5:          mlr3torch,torch
#>  6:          mlr3torch,torch
#>  7:          mlr3torch,torch
#>  8:          mlr3torch,torch
#>  9: progress,mlr3torch,torch
#> 10: tfevents,mlr3torch,torch
#> 11:          mlr3torch,torch
```
