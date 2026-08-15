# Loss Configuration

Configures the loss of a deep learning model.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `shapes_out()`.

## Parameters

The parameters are defined dynamically from the loss set during
construction.

## Internals

During training the loss is cloned and added to the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

## See also

Other Model Configuration:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`mlr_pipeops_torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_callbacks.md),
[`mlr_pipeops_torch_optimizer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_optimizer.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpTorchLoss`

## Methods

### Public methods

- [`PipeOpTorchLoss$new()`](#method-PipeOpTorchLoss-initialize)

- [`PipeOpTorchLoss$clone()`](#method-PipeOpTorchLoss-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchLoss$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchLoss$new(loss, id = "torch_loss", param_vals = list())

#### Arguments

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md)
  or `character(1)` or `nn_loss`)  
  The loss (or something convertible via
  [`as_torch_loss()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_loss.md)).

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchLoss$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchLoss$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
po_loss = po("torch_loss", loss = t_loss("cross_entropy"))
po_loss$param_set
#> <ParamSet(3)>
#>              id    class lower upper nlevels default  value
#>          <char>   <char> <num> <num>   <num>  <list> <list>
#> 1: class_weight ParamUty    NA    NA     Inf  [NULL] [NULL]
#> 2: ignore_index ParamInt  -Inf   Inf     Inf    -100 [NULL]
#> 3:    reduction ParamFct    NA    NA       2    mean [NULL]
mdin = po("torch_ingress_num")$train(list(tsk("iris")))
mdin[[1L]]$loss
#> NULL
mdout = po_loss$train(mdin)[[1L]]
mdout$loss
#> <TorchLoss:cross_entropy> Cross Entropy
#> * Generator: function
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: classif
```
