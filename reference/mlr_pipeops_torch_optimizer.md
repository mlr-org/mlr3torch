# Optimizer Configuration

Configures the optimizer of a deep learning model.

## Parameters

The parameters are defined dynamically from the optimizer that is set
during construction.

## Input and Output Channels

There is one input channel `"input"` and one output channel `"output"`.
During *training*, the channels are of class
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).
During *prediction*, the channels are of class
[`Task`](https://mlr3.mlr-org.com/reference/Task.html).

## State

The state is the value calculated by the public method `shapes_out()`.

## Internals

During training, the optimizer is cloned and added to the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).
Note that the parameter set of the stored
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md)
is reference-identical to the parameter set of the pipeop itself.

## See also

Other PipeOp:
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_callbacks.md)

Other Model Configuration:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`mlr_pipeops_torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_callbacks.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_loss.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpTorchOptimizer`

## Methods

### Public methods

- [`PipeOpTorchOptimizer$new()`](#method-PipeOpTorchOptimizer-new)

- [`PipeOpTorchOptimizer$clone()`](#method-PipeOpTorchOptimizer-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchOptimizer$new(
      optimizer = t_opt("adam"),
      id = "torch_optimizer",
      param_vals = list()
    )

#### Arguments

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md)
  or `character(1)` or `torch_optimizer_generator`)  
  The optimizer (or something convertible via
  [`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/reference/as_torch_optimizer.md)).

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchOptimizer$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
po_opt = po("torch_optimizer", "sgd", lr = 0.01)
po_opt$param_set
#> <ParamSet(6)>
#>              id    class lower upper nlevels        default  value
#>          <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:           lr ParamDbl     0   Inf     Inf <NoDefault[0]>   0.01
#> 2:     momentum ParamDbl     0     1     Inf              0 [NULL]
#> 3:    dampening ParamDbl     0     1     Inf              0 [NULL]
#> 4: weight_decay ParamDbl     0     1     Inf              0 [NULL]
#> 5:     nesterov ParamLgl    NA    NA       2          FALSE [NULL]
#> 6: param_groups ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
mdin = po("torch_ingress_num")$train(list(tsk("iris")))
mdin[[1L]]$optimizer
#> NULL
mdout = po_opt$train(mdin)
mdout[[1L]]$optimizer
#> <TorchOptimizer:sgd> Stochastic Gradient Descent
#> * Generator: optim_ignite_sgd
#> * Parameters: lr=0.01
#> * Packages: torch,mlr3torch
```
