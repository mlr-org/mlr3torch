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
