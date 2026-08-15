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

## Examples

``` r
# Define a custom loss, here the quantile (pinball) loss for quantile regression.
# The class must contain "nn_loss", as this is what as_torch_loss() dispatches on.
nn_quantile_loss = nn_module(c("nn_quantile_loss", "nn_loss"),
  initialize = function(q = 0.5) {
    self$q = q
  },
  forward = function(input, target) {
    d = target - input
    torch_mean(torch_max(self$q * d, other = (self$q - 1) * d))
  }
)

# additional arguments are passed to the TorchLoss constructor
quantile_loss = as_torch_loss(nn_quantile_loss, task_types = "regr")
quantile_loss
#> <TorchLoss:nn_quantile_loss> nn_quantile_loss
#> * Generator: nn_quantile_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr
# the parameters are inferred from the loss' initialize method
quantile_loss$param_set
#> <ParamSet(1)>
#>        id    class lower upper nlevels        default  value
#>    <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:      q ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
# and can be configured when using the loss in a learner
lrn("regr.mlp", loss = quantile_loss, loss.q = 0.9)
#> 
#> ── <LearnerTorchMLP> (regr.mlp): Multi Layer Perceptron ────────────────────────
#> • Model: -
#> • Parameters: device=auto, num_threads=1, seed=random, eval_freq=1,
#> measures_train=<list>, measures_valid=<list>, patience=0, min_delta=0,
#> restore_best_weights=FALSE, shuffle=TRUE, tensor_dataset=FALSE,
#> jit_trace=FALSE, neurons=integer(0), p=0.1, activation=<nn_relu>,
#> activation_args=<list>, loss.q=0.9
#> • Validate: NULL
#> • Packages: mlr3, mlr3torch, and torch
#> • Predict Types: [response]
#> • Feature Types: integer, numeric, and lazy_tensor
#> • Encapsulation: none (fallback: -)
#> • Properties: internal_tuning, marshal, and validation
#> • Other settings: use_weights = 'error', predict_raw = 'FALSE'
#> • Optimizer: adam
#> • Loss: nn_quantile_loss
#> • Callbacks: -

# predefined losses can be converted from their key,
# this is the same as t_loss("mse")
as_torch_loss("mse")
#> <TorchLoss:mse> Mean Squared Error
#> * Generator: nn_mse_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr

# TorchLosses are returned as-is, unless clone is TRUE
loss = t_loss("mse")
identical(as_torch_loss(loss), loss)
#> [1] TRUE
identical(as_torch_loss(loss, clone = TRUE), loss)
#> [1] FALSE
```
