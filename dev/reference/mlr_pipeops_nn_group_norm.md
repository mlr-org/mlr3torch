# Group Normalization

Applies Group Normalization for last certain number of dimensions.

## nn_module

Calls
[`torch::nn_group_norm()`](https://torch.mlverse.org/docs/reference/nn_group_norm.html)
when trained. The parameter `num_channels` is inferred as the second
dimension of the input shape.

## Parameters

- `num_groups` :: `integer(1)`  
  The number of groups to separate the channels into. Must divide the
  number of channels. Setting it to `1` normalizes over all channels at
  once (layer normalization), setting it to the number of channels
  normalizes each channel on its own (instance normalization).

- `eps` :: `numeric(1)`  
  A value added to the denominator for numerical stability. Default:
  `1e-5`.

- `affine` :: `logical(1)`  
  Whether to learn per-channel affine parameters initialized to `1` for
  weights and to `0` for biases. Default: `TRUE`.

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
-\> `PipeOpTorchGroupNorm`

## Methods

### Public methods

- [`PipeOpTorchGroupNorm$new()`](#method-PipeOpTorchGroupNorm-initialize)

- [`PipeOpTorchGroupNorm$clone()`](#method-PipeOpTorchGroupNorm-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchGroupNorm$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchGroupNorm$new(id = "nn_group_norm", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchGroupNorm$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchGroupNorm$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("group_norm", num_groups = 1)
pipeop
#> 
#> ── PipeOp <group_norm>: not trained ────────────────────────────────────────────
#> Values: num_groups=1
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
#>            id    class lower upper nlevels        default  value
#>        <char>   <char> <num> <num>   <num>         <list> <list>
#> 1: num_groups ParamInt     1   Inf     Inf <NoDefault[0]>      1
#> 2:        eps ParamDbl     0   Inf     Inf          1e-05 [NULL]
#> 3:     affine ParamLgl    NA    NA       2           TRUE [NULL]
```
