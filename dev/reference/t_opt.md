# Optimizers Quick Access

Retrieves one or more
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md)s
from
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md).
Works like
[`mlr3::lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html) and
[`mlr3::lrns()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html).

## Usage

``` r
t_opt(.key, ...)

t_opts(.keys, ...)
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
  The keys of the optimizers.

## Value

A
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md)

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
[`t_loss()`](https://mlr3torch.mlr-org.com/dev/reference/t_loss.md)

Other Dictionary:
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md)

## Examples

``` r
t_opt("adam", lr = 0.1)
#> <TorchOptimizer:adam> Adaptive Moment Estimation
#> * Generator: optim_ignite_adam
#> * Parameters: lr=0.1
#> * Packages: torch,mlr3torch
# get the dictionary
t_opt()
#> <DictionaryMlr3torchOptimizers> with 5 stored values
#> Keys: adagrad, adam, adamw, rmsprop, sgd
t_opts(c("adam", "sgd"))
#> $adam
#> <TorchOptimizer:adam> Adaptive Moment Estimation
#> * Generator: optim_ignite_adam
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> 
#> $sgd
#> <TorchOptimizer:sgd> Stochastic Gradient Descent
#> * Generator: optim_ignite_sgd
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> 
# get the dictionary
t_opts()
#> <DictionaryMlr3torchOptimizers> with 5 stored values
#> Keys: adagrad, adam, adamw, rmsprop, sgd
```
