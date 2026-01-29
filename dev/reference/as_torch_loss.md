# Convert to TorchLoss

Converts an object to a
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md).

## Usage

``` r
as_torch_loss(x, clone = FALSE, ...)
```

## Arguments

- x:

  (any)  
  Object to convert to a
  [`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md).

- clone:

  (`logical(1)`)  
  Whether to make a deep clone.

- ...:

  (any)  
  Additional arguments. Currently used to pass additional constructor
  arguments to
  [`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md)
  for objects of type `nn_loss`.

## Value

[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md).

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/dev/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/dev/reference/t_opt.md)
