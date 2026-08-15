# Hard Tanh Activation Function

Applies the HardTanh function element-wise.

## nn_module

Calls
[`torch::nn_hardtanh()`](https://torch.mlverse.org/docs/reference/nn_hardtanh.html)
when trained.

## Parameters

- `min_val` :: `numeric(1)`  
  Minimum value of the linear region range. Default: -1.

- `max_val` :: `numeric(1)`  
  Maximum value of the linear region range. Default: 1.

- `inplace` :: `logical(1)`  
  Can optionally do the operation in-place. Default: `FALSE`.

## State

The state is the value calculated by the public method `$shapes_out()`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchHardTanh`

## Methods

### Public methods

- [`PipeOpTorchHardTanh$new()`](#method-PipeOpTorchHardTanh-initialize)

- [`PipeOpTorchHardTanh$clone()`](#method-PipeOpTorchHardTanh-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchHardTanh$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchHardTanh$new(id = "nn_hardtanh", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchHardTanh$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchHardTanh$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("hardtanh")
pipeop
#> 
#> ── PipeOp <hardtanh>: not trained ──────────────────────────────────────────────
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
#> <ParamSet(3)>
#>         id    class lower upper nlevels default  value
#>     <char>   <char> <num> <num>   <num>  <list> <list>
#> 1: min_val ParamDbl  -Inf   Inf     Inf      -1 [NULL]
#> 2: max_val ParamDbl  -Inf   Inf     Inf       1 [NULL]
#> 3: inplace ParamLgl    NA    NA       2   FALSE [NULL]
```
