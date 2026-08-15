# SoftPlus Activation Function

Applies element-wise, the function \\Softplus(x) = 1/\beta \* log(1 +
exp(\beta \* x))\\.

## nn_module

Calls
[`torch::nn_softplus()`](https://torch.mlverse.org/docs/reference/nn_softplus.html)
when trained.

## Parameters

- `beta` :: `numeric(1)`  
  The beta value for the Softplus formulation. Default: 1

- `threshold` :: `numeric(1)`  
  Values above this revert to a linear function. Default: 20

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchSoftPlus`

## Methods

### Public methods

- [`PipeOpTorchSoftPlus$new()`](#method-PipeOpTorchSoftPlus-initialize)

- [`PipeOpTorchSoftPlus$clone()`](#method-PipeOpTorchSoftPlus-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchSoftPlus$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchSoftPlus$new(id = "nn_softplus", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchSoftPlus$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchSoftPlus$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("softplus")
pipeop
#> 
#> ── PipeOp <softplus>: not trained ──────────────────────────────────────────────
#> Values: list()
#> 
#> ── Input channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>   input ModelDescriptor    Task
#> 
#> ── Output channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>  output ModelDescriptor    Task
# The available parameters
pipeop$param_set
#> <ParamSet(2)>
#>           id    class lower upper nlevels default  value
#>       <char>   <char> <num> <num>   <num>  <list> <list>
#> 1:      beta ParamDbl  -Inf   Inf     Inf       1 [NULL]
#> 2: threshold ParamDbl  -Inf   Inf     Inf      20 [NULL]
```
