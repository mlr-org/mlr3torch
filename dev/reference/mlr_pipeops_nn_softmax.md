# Softmax

Applies a softmax function.

## nn_module

Calls
[`torch::nn_softmax()`](https://torch.mlverse.org/docs/reference/nn_softmax.html)
when trained.

## Parameters

- `dim` :: `integer(1)`  
  A dimension along which Softmax will be computed (so every slice along
  dim will sum to 1).

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
-\> `PipeOpTorchSoftmax`

## Methods

### Public methods

- [`PipeOpTorchSoftmax$new()`](#method-PipeOpTorchSoftmax-initialize)

- [`PipeOpTorchSoftmax$clone()`](#method-PipeOpTorchSoftmax-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchSoftmax$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchSoftmax$new(id = "nn_softmax", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchSoftmax$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchSoftmax$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("softmax")
pipeop
#> 
#> ── PipeOp <softmax>: not trained ───────────────────────────────────────────────
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
#>        id    class lower upper nlevels        default  value
#>    <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:    dim ParamInt  -Inf   Inf     Inf <NoDefault[0]> [NULL]
```
