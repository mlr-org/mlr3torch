# Loss Functions

Dictionary of torch loss descriptors. See
[`t_loss()`](https://mlr3torch.mlr-org.com/reference/t_loss.md) for
conveniently retrieving a loss function. Can be converted to a
[`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html) using
[`as.data.table`](https://rdrr.io/pkg/data.table/man/as.data.table.html).

## Usage

``` r
mlr3torch_losses
```

## Format

An object of class `DictionaryMlr3torchLosses` (inherits from
`Dictionary`, `R6`) of length 12.

## Available Loss Functions

cross_entropy, l1, mse

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/reference/as_torch_optimizer.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

Other Dictionary:
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/reference/mlr3torch_optimizers.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

## Examples

``` r
mlr3torch_losses$get("mse")
#> <TorchLoss:mse> Mean Squared Error
#> * Generator: nn_mse_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr
# is equivalent to
t_loss("mse")
#> <TorchLoss:mse> Mean Squared Error
#> * Generator: nn_mse_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr
# convert to a data.table
as.data.table(mlr3torch_losses)
#> Key: <key>
#>              key              label task_types        packages
#>           <char>             <char>     <list>          <list>
#> 1: cross_entropy      Cross Entropy    classif torch,mlr3torch
#> 2:            l1     Absolute Error       regr torch,mlr3torch
#> 3:           mse Mean Squared Error       regr torch,mlr3torch
```
