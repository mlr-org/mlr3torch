# Unflattens a Tensor

Unflattens a tensor dim expanding it to a desired shape. For use with
\[[nn_sequential](https://torch.mlverse.org/docs/reference/nn_sequential.html).

## nn_module

Calls
[`torch::nn_unflatten()`](https://torch.mlverse.org/docs/reference/nn_unflatten.html)
when trained.

## Parameters

- `dim` :: `integer(1)`  
  The dimension to unflatten. Negative values are interpreted downwards
  from the last dimension.

- `unflattened_size` ::
  [`integer()`](https://rdrr.io/r/base/integer.html)  
  The sizes that replace that dimension. One of them at most can be
  `-1`, which torch infers from the size of the dimension being
  unflattened.

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
-\> `PipeOpTorchUnflatten`

## Methods

### Public methods

- [`PipeOpTorchUnflatten$new()`](#method-PipeOpTorchUnflatten-initialize)

- [`PipeOpTorchUnflatten$clone()`](#method-PipeOpTorchUnflatten-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchUnflatten$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchUnflatten$new(id = "nn_unflatten", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchUnflatten$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchUnflatten$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("unflatten", dim = 2, unflattened_size = c(2, 2))
pipeop
#> 
#> ── PipeOp <unflatten>: not trained ─────────────────────────────────────────────
#> Values: dim=2, unflattened_size=2,2
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
#>                  id    class lower upper nlevels        default  value
#>              <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:              dim ParamInt  -Inf   Inf     Inf <NoDefault[0]>      2
#> 2: unflattened_size ParamUty    NA    NA     Inf <NoDefault[0]>    2,2
```
