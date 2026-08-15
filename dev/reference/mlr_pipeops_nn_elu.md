# ELU Activation Function

Applies element-wise, \$\$ELU(x) = max(0,x) + min(0, \alpha \* (exp(x) -
1))\$\$.

## nn_module

Calls
[`torch::nn_elu()`](https://torch.mlverse.org/docs/reference/nn_elu.html)
when trained.

## Parameters

- `alpha` :: `numeric(1)`  
  The alpha value for the ELU formulation. Default: 1.0

- `inplace` :: `logical(1)`  
  Whether to do the operation in-place. Default: `FALSE`.

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
-\> `PipeOpTorchELU`

## Methods

### Public methods

- [`PipeOpTorchELU$new()`](#method-PipeOpTorchELU-initialize)

- [`PipeOpTorchELU$clone()`](#method-PipeOpTorchELU-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchELU$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchELU$new(id = "nn_elu", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchELU$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchELU$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("elu")
pipeop
#> 
#> ── PipeOp <elu>: not trained ───────────────────────────────────────────────────
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
#>         id    class lower upper nlevels default  value
#>     <char>   <char> <num> <num>   <num>  <list> <list>
#> 1:   alpha ParamDbl  -Inf   Inf     Inf       1 [NULL]
#> 2: inplace ParamLgl    NA    NA       2   FALSE [NULL]
```
