# Sugar Function for Torch Callback

Retrieves one or more
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)s
from
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md).
Works like
[`mlr3::lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html) and
[`mlr3::lrns()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html).

## Usage

``` r
t_clbk(.key, ...)

t_clbks(.keys)
```

## Arguments

- .key:

  (`character(1)`)  
  The key of the torch callback.

- ...:

  (any)  
  See description of
  [`dictionary_sugar_get()`](https://mlr3misc.mlr-org.com/reference/dictionary_sugar_get.html).

- .keys:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The keys of the callbacks.

## Value

[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)

[`list()`](https://rdrr.io/r/base/list.html) of
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)s

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/reference/mlr3torch_optimizers.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

## Examples

``` r
t_clbk("progress")
#> <TorchCallback:progress> Progress
#> * Generator: CallbackSetProgress
#> * Parameters: list()
#> * Packages: progress,mlr3torch,torch
```
