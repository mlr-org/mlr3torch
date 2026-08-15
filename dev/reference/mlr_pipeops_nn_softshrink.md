# Soft Shrink Activation Function

Applies the soft shrinkage function elementwise

## nn_module

Calls
[`torch::nn_softshrink()`](https://torch.mlverse.org/docs/reference/nn_softshrink.html)
when trained.

## Parameters

- `lambd` :: `numeric(1)`  
  The lambda (must be no less than zero) value for the Softshrink
  formulation. Default: 0.5

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
-\> `PipeOpTorchSoftShrink`

## Methods

### Public methods

- [`PipeOpTorchSoftShrink$new()`](#method-PipeOpTorchSoftShrink-initialize)

- [`PipeOpTorchSoftShrink$clone()`](#method-PipeOpTorchSoftShrink-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchSoftShrink$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchSoftShrink$new(id = "nn_softshrink", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchSoftShrink$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchSoftShrink$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("softshrink")
pipeop
#> 
#> ── PipeOp <softshrink>: not trained ────────────────────────────────────────────
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
#> <ParamSet(1)>
#>        id    class lower upper nlevels default  value
#>    <char>   <char> <num> <num>   <num>  <list> <list>
#> 1:  lambd ParamDbl     0   Inf     Inf     0.5 [NULL]
```
