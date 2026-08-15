# Linear Layer

Applies a linear transformation to the incoming data: \\y = xA^T + b\\.

## nn_module

Calls
[`torch::nn_linear()`](https://torch.mlverse.org/docs/reference/nn_linear.html)
when trained where the parameter `in_features` is inferred as the second
to last dimension of the input tensor.

## Parameters

- `out_features` :: `integer(1)`  
  The output features of the linear layer.

- `bias` :: `logical(1)`  
  Whether to use a bias. Default is `TRUE`.

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
-\> `PipeOpTorchLinear`

## Methods

### Public methods

- [`PipeOpTorchLinear$new()`](#method-PipeOpTorchLinear-initialize)

- [`PipeOpTorchLinear$clone()`](#method-PipeOpTorchLinear-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchLinear$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchLinear$new(id = "nn_linear", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchLinear$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchLinear$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("linear", out_features = 10)
pipeop
#> 
#> ── PipeOp <linear>: not trained ────────────────────────────────────────────────
#> Values: out_features=10
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
#>              id    class lower upper nlevels        default  value
#>          <char>   <char> <num> <num>   <num>         <list> <list>
#> 1: out_features ParamInt     1   Inf     Inf <NoDefault[0]>     10
#> 2:         bias ParamLgl    NA    NA       2           TRUE [NULL]
```
