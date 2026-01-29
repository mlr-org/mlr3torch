# Optimizers

Dictionary of torch optimizers. Use
[`t_opt`](https://mlr3torch.mlr-org.com/reference/t_opt.md) for
conveniently retrieving optimizers. Can be converted to a
[`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html) using
[`as.data.table`](https://rdrr.io/pkg/data.table/man/as.data.table.html).

## Usage

``` r
mlr3torch_optimizers
```

## Format

An object of class `DictionaryMlr3torchOptimizers` (inherits from
`Dictionary`, `R6`) of length 12.

## Available Optimizers

adagrad, adam, adamw, rmsprop, sgd

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/reference/mlr3torch_losses.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

Other Dictionary:
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/reference/mlr3torch_losses.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

## Examples

``` r
mlr3torch_optimizers$get("adam")
#> <TorchOptimizer:adam> Adaptive Moment Estimation
#> * Generator: optim_ignite_adam
#> * Parameters: list()
#> * Packages: torch,mlr3torch
# is equivalent to
t_opt("adam")
#> <TorchOptimizer:adam> Adaptive Moment Estimation
#> * Generator: optim_ignite_adam
#> * Parameters: list()
#> * Packages: torch,mlr3torch
# convert to a data.table
as.data.table(mlr3torch_optimizers)
#> Key: <key>
#>        key                                 label        packages
#>     <char>                                <char>          <list>
#> 1: adagrad           Adaptive Gradient algorithm torch,mlr3torch
#> 2:    adam            Adaptive Moment Estimation torch,mlr3torch
#> 3:   adamw Decoupled Weight Decay Regularization torch,mlr3torch
#> 4: rmsprop          Root Mean Square Propagation torch,mlr3torch
#> 5:     sgd           Stochastic Gradient Descent torch,mlr3torch
```
