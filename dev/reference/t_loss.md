# Loss Function Quick Access

Retrieve one or more
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md)es
from
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md).
Works like
[`mlr3::lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html) and
[`mlr3::lrns()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html).

## Usage

``` r
t_loss(.key, ...)

t_losses(.keys, ...)
```

## Arguments

- .key:

  (`character(1)`)  
  Key of the object to retrieve.

- ...:

  (any)  
  See description of
  [`dictionary_sugar_get`](https://mlr3misc.mlr-org.com/reference/dictionary_sugar_get.html).

- .keys:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The keys of the losses.

## Value

A
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md)

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/dev/reference/t_opt.md)

## Examples

``` r
t_loss("mse", reduction = "mean")
#> <TorchLoss:mse> Mean Squared Error
#> * Generator: nn_mse_loss
#> * Parameters: reduction=mean
#> * Packages: torch,mlr3torch
#> * Task Types: regr
# get the dictionary
t_loss()
#> <DictionaryMlr3torchLosses> with 3 stored values
#> Keys: cross_entropy, l1, mse
t_losses(c("mse", "l1"))
#> $mse
#> <TorchLoss:mse> Mean Squared Error
#> * Generator: nn_mse_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr
#> 
#> $l1
#> <TorchLoss:l1> Absolute Error
#> * Generator: nn_l1_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr
#> 
# get the dictionary
t_losses()
#> <DictionaryMlr3torchLosses> with 3 stored values
#> Keys: cross_entropy, l1, mse
```
