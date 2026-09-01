# Threshold Activation Function

Thresholds each element of the input Tensor.

## nn_module

Calls
[`torch::nn_threshold()`](https://torch.mlverse.org/docs/reference/nn_threshold.html)
when trained.

## Parameters

- `threshold` :: `numeric(1)`  
  The value to threshold at.

- `value` :: `numeric(1)`  
  The value to replace with.

- `inplace` :: `logical(1)`  
  Can optionally do the operation in-place. Default: `FALSE`.

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
-\> `PipeOpTorchThreshold`

## Methods

### Public methods

- [`PipeOpTorchThreshold$new()`](#method-PipeOpTorchThreshold-initialize)

- [`PipeOpTorchThreshold$clone()`](#method-PipeOpTorchThreshold-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchThreshold$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchThreshold$new(id = "nn_threshold", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchThreshold$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchThreshold$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("threshold", threshold = 1, value = 2)
pipeop
#> 
#> ── PipeOp <threshold>: not trained ─────────────────────────────────────────────
#> Values: threshold=1, value=2
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
#> <ParamSet(3)>
#>           id    class lower upper nlevels        default  value
#>       <char>   <char> <num> <num>   <num>         <list> <list>
#> 1: threshold ParamDbl  -Inf   Inf     Inf <NoDefault[0]>      1
#> 2:     value ParamDbl  -Inf   Inf     Inf <NoDefault[0]>      2
#> 3:   inplace ParamLgl    NA    NA       2          FALSE [NULL]
```
