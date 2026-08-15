# Reshape a Tensor

Reshape a tensor to the given shape.

## nn_module

Calls
[`nn_reshape()`](https://mlr3torch.mlr-org.com/dev/reference/nn_reshape.md)
when trained. This internally calls
[`torch::torch_reshape()`](https://torch.mlverse.org/docs/reference/torch_reshape.html)
with the given `shape`.

## Parameters

- `shape` :: [`integer()`](https://rdrr.io/r/base/integer.html) \|
  `function()`  
  The desired output shape. One dimension at most can be `-1`, which
  torch infers from the number of elements. The first dimension is the
  batch dimension.

  It can also be a `function(shape)` that is called on the input shape
  and returns the output shape, e.g. `\(shape) c(shape[1:2], 10)`. This
  expresses a reshape for inputs whose sizes are not known in advance,
  because the function is called again on the shape of the actual tensor
  when the network runs. Note that it is called with a shape that can
  contain `NA`s during shape inference. This is e.g. useful when there
  are multiple unknown dimensions such as `(batch, sequence, ...)`.

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
-\> `PipeOpTorchReshape`

## Methods

### Public methods

- [`PipeOpTorchReshape$new()`](#method-PipeOpTorchReshape-initialize)

- [`PipeOpTorchReshape$clone()`](#method-PipeOpTorchReshape-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchReshape$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchReshape$new(id = "nn_reshape", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchReshape$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchReshape$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("reshape")
pipeop
#> 
#> ── PipeOp <reshape>: not trained ───────────────────────────────────────────────
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
#> 1:  shape ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
```
