# PReLU Activation Function

Applies element-wise the function \\PReLU(x) = max(0,x) + weight \*
min(0,x)\\ where weight is a learnable parameter.

## nn_module

Calls
[`torch::nn_prelu()`](https://torch.mlverse.org/docs/reference/nn_prelu.html)
when trained.

## Parameters

- `num_parameters` :: `integer(1)`  
  Number of `a` parameters to learn. Although it takes an integer as
  input, only two values are legitimate: `1`, or the number of channels
  of the input. Default: 1.

- `init` :: `numeric(1)`  
  The initial value of `a`. Default: 0.25.

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
-\> `PipeOpTorchPReLU`

## Methods

### Public methods

- [`PipeOpTorchPReLU$new()`](#method-PipeOpTorchPReLU-initialize)

- [`PipeOpTorchPReLU$clone()`](#method-PipeOpTorchPReLU-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchPReLU$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchPReLU$new(id = "nn_prelu", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchPReLU$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchPReLU$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("prelu")
pipeop
#> 
#> ── PipeOp <prelu>: not trained ─────────────────────────────────────────────────
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
#>                id    class lower upper nlevels default  value
#>            <char>   <char> <num> <num>   <num>  <list> <list>
#> 1: num_parameters ParamInt     1   Inf     Inf       1 [NULL]
#> 2:           init ParamDbl  -Inf   Inf     Inf    0.25 [NULL]
```
