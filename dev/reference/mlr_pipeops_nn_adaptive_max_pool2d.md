# 2D Adaptive Max Pooling

Applies a 2D adaptive max pooling over an input signal composed of
several input planes.

## nn_module

Calls
[`nn_adaptive_max_pool2d()`](https://torch.mlverse.org/docs/reference/nn_adaptive_max_pool2d.html)
during training.

## Parameters

- `output_size` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The target output size. Can be a single number or a vector.

## State

The state is the value calculated by the public method `$shapes_out()`.

## Input and Output Channels

If `return_indices` is `FALSE` during construction, there is one input
channel 'input' and one output channel 'output'. If `return_indices` is
`TRUE`, there are two output channels 'output' and 'indices'. For an
explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchAdaptiveMaxPool` -\> `PipeOpTorchAdaptiveMaxPool2D`

## Methods

### Public methods

- [`PipeOpTorchAdaptiveMaxPool2D$new()`](#method-PipeOpTorchAdaptiveMaxPool2D-initialize)

- [`PipeOpTorchAdaptiveMaxPool2D$clone()`](#method-PipeOpTorchAdaptiveMaxPool2D-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchAdaptiveMaxPool2D$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchAdaptiveMaxPool2D$new(
      id = "nn_adaptive_max_pool2d",
      return_indices = FALSE,
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `return_indices`:

  (`logical(1)`)  
  Whether to return the indices. If this is `TRUE`, there are two output
  channels `"output"` and `"indices"`.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchAdaptiveMaxPool2D$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchAdaptiveMaxPool2D$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("adaptive_max_pool2d")
pipeop
#> 
#> ── PipeOp <adaptive_max_pool2d>: not trained ───────────────────────────────────
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
