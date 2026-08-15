# 2D Average Pooling

Applies 2D average-pooling operation in \\kH \* kW\\ regions by step
size \\sH \* sW\\ steps. The number of output features is equal to the
number of input planes.

## nn_module

Calls
[`nn_avg_pool2d()`](https://torch.mlverse.org/docs/reference/nn_avg_pool2d.html)
during training.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Parameters

- `kernel_size` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The size of the window. Can be a single number or a vector.

- `stride` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The stride of the window. Can be a single number or a vector. Default:
  `kernel_size`.

- `padding` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Implicit zero paddings on both sides of the input. Can be a single
  number or a vector. Default: 0.

- `ceil_mode` :: `logical(1)`  
  When `TRUE`, will use ceil instead of floor to compute the output
  shape. Default: `FALSE`.

- `count_include_pad` :: `logical(1)`  
  When `TRUE`, will include the zero-padding in the averaging
  calculation. Default: `TRUE`.

- `divisor_override` :: `numeric(1)`  
  If specified, it will be used as divisor, otherwise size of the
  pooling region will be used. Default: NULL. Only available for
  `nn_avg_pool2d` and `nn_avg_pool3d`, i.e. this parameter does not
  exist for `nn_avg_pool1d`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchAvgPool` -\> `PipeOpTorchAvgPool2D`

## Methods

### Public methods

- [`PipeOpTorchAvgPool2D$new()`](#method-PipeOpTorchAvgPool2D-initialize)

- [`PipeOpTorchAvgPool2D$clone()`](#method-PipeOpTorchAvgPool2D-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchAvgPool2D$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchAvgPool2D$new(id = "nn_avg_pool2d", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchAvgPool2D$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchAvgPool2D$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("avg_pool2d")
pipeop
#> 
#> ── PipeOp <avg_pool2d>: not trained ────────────────────────────────────────────
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
#> <ParamSet(6)>
#>                   id    class lower upper nlevels        default  value
#>               <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:       kernel_size ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:            stride ParamUty    NA    NA     Inf         [NULL] [NULL]
#> 3:           padding ParamUty    NA    NA     Inf              0 [NULL]
#> 4:         ceil_mode ParamLgl    NA    NA       2          FALSE [NULL]
#> 5: count_include_pad ParamLgl    NA    NA       2           TRUE [NULL]
#> 6:  divisor_override ParamDbl     0   Inf     Inf         [NULL] [NULL]
```
