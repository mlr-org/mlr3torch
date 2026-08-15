# 2D Adaptive Average Pooling

Applies a 2D adaptive average pooling over an input signal composed of
several input planes.

## nn_module

Calls
[`nn_adaptive_avg_pool2d()`](https://torch.mlverse.org/docs/reference/nn_adaptive_avg_pool2d.html)
during training.

## Parameters

- `output_size` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The target output size. Can be a single number or a vector.

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
-\> `PipeOpTorchAdaptiveAvgPool` -\> `PipeOpTorchAdaptiveAvgPool2D`

## Methods

### Public methods

- [`PipeOpTorchAdaptiveAvgPool2D$new()`](#method-PipeOpTorchAdaptiveAvgPool2D-initialize)

- [`PipeOpTorchAdaptiveAvgPool2D$clone()`](#method-PipeOpTorchAdaptiveAvgPool2D-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchAdaptiveAvgPool2D$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchAdaptiveAvgPool2D$new(
      id = "nn_adaptive_avg_pool2d",
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchAdaptiveAvgPool2D$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchAdaptiveAvgPool2D$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("adaptive_avg_pool2d")
pipeop
#> 
#> ── PipeOp <adaptive_avg_pool2d>: not trained ───────────────────────────────────
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
#>             id    class lower upper nlevels        default  value
#>         <char>   <char> <num> <num>   <num>         <list> <list>
#> 1: output_size ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
```
