# Convert to TorchOptimizer

Converts an object to a
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md).

## Usage

``` r
as_torch_optimizer(x, clone = FALSE, ...)
```

## Arguments

- x:

  (any)  
  Object to convert to a
  [`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md).

- clone:

  (`logical(1)`)  
  Whether to make a deep clone. Default is `FALSE`.

- ...:

  (any)  
  Additional arguments. Currently used to pass additional constructor
  arguments to
  [`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md)
  for objects of type `torch_optimizer_generator`.

## Value

[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md)

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_loss.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/dev/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/dev/reference/t_opt.md)

## Examples

``` r
# convert a `torch::torch_optimizer_generator`
as_torch_optimizer(optim_adamw)
#> <TorchOptimizer:optim_adamw> optim_adamw
#> * Generator: optim_adamw
#> * Parameters: list()
#> * Packages: torch,mlr3torch
# the id defaults to the name of the generator, but can be overwritten
as_torch_optimizer(optim_adamw, id = "my_adamw", label = "My AdamW")
#> <TorchOptimizer:my_adamw> My AdamW
#> * Generator: optim_adamw
#> * Parameters: list()
#> * Packages: torch,mlr3torch

# convert a key of mlr3torch_optimizers, this is the same as t_opt("adamw")
as_torch_optimizer("adamw")
#> <TorchOptimizer:adamw> Decoupled Weight Decay Regularization
#> * Generator: optim_ignite_adamw
#> * Parameters: list()
#> * Packages: torch,mlr3torch

# TorchOptimizers are returned as-is, unless clone is TRUE
opt = t_opt("adamw")
identical(as_torch_optimizer(opt), opt)
#> [1] TRUE
identical(as_torch_optimizer(opt, clone = TRUE), opt)
#> [1] FALSE
```
