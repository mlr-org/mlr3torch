# Layer Normalization

Applies Layer Normalization for last certain number of dimensions.

## nn_module

Calls
[`torch::nn_layer_norm()`](https://torch.mlverse.org/docs/reference/nn_layer_norm.html)
when trained. The parameter `normalized_shape` is inferred as the
dimensions of the last `dims` dimensions of the input shape.

## Parameters

- `dims` :: `integer(1)`  
  The number of dimensions over which will be normalized (starting from
  the last dimension).

- `elementwise_affine` :: `logical(1)`  
  Whether to learn affine-linear parameters initialized to `1` for
  weights and to `0` for biases. The default is `TRUE`.

- `eps` :: `numeric(1)`  
  A value added to the denominator for numerical stability.

## State

The state is the value calculated by the public method `$shapes_out()`.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchLayerNorm`

## Methods

### Public methods

- [`PipeOpTorchLayerNorm$new()`](#method-PipeOpTorchLayerNorm-initialize)

- [`PipeOpTorchLayerNorm$clone()`](#method-PipeOpTorchLayerNorm-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchLayerNorm$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchLayerNorm$new(id = "nn_layer_norm", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchLayerNorm$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchLayerNorm$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("layer_norm", dims = 1)
pipeop
#> 
#> ── PipeOp <layer_norm>: not trained ────────────────────────────────────────────
#> Values: dims=1
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
#>                    id    class lower upper nlevels        default  value
#>                <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:               dims ParamInt     1   Inf     Inf <NoDefault[0]>      1
#> 2: elementwise_affine ParamLgl    NA    NA       2           TRUE [NULL]
#> 3:                eps ParamDbl     0   Inf     Inf          1e-05 [NULL]
```
