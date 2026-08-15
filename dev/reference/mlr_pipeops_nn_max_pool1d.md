# 1D Max Pooling

Applies a 1D max pooling over an input signal composed of several input
planes.

## nn_module

Calls
[`torch::nn_max_pool1d()`](https://torch.mlverse.org/docs/reference/nn_max_pool1d.html)
during training.

## Parameters

- `kernel_size` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The size of the window. Can be single number or a vector.

- `stride` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The stride of the window. Can be a single number or a vector. Default:
  `kernel_size`

- `padding` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Implicit zero paddings on both sides of the input. Can be a single
  number or a tuple (padW,). Default: 0

- `dilation` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Controls the spacing between the kernel points; also known as the a
  trous algorithm. Default: 1

- `ceil_mode` :: `logical(1)`  
  When True, will use ceil instead of floor to compute the output shape.
  Default: `FALSE`

## Input and Output Channels

If `return_indices` is `FALSE` during construction, there is one input
channel 'input' and one output channel 'output'. If `return_indices` is
`TRUE`, there are two output channels 'output' and 'indices'. For an
explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchMaxPool` -\> `PipeOpTorchMaxPool1D`

## Methods

### Public methods

- [`PipeOpTorchMaxPool1D$new()`](#method-PipeOpTorchMaxPool1D-initialize)

- [`PipeOpTorchMaxPool1D$clone()`](#method-PipeOpTorchMaxPool1D-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchMaxPool1D$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchMaxPool1D$new(
      id = "nn_max_pool1d",
      return_indices = FALSE,
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `return_indices`:

  (`logical(1)`)  
  Whether to return the indices. If this is `TRUE`, there are two output
  channels `"output"` and `"indices"`.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchMaxPool1D$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchMaxPool1D$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("max_pool1d")
pipeop
#> 
#> ── PipeOp <max_pool1d>: not trained ────────────────────────────────────────────
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
#> <ParamSet(5)>
#>             id    class lower upper nlevels        default  value
#>         <char>   <char> <num> <num>   <num>         <list> <list>
#> 1: kernel_size ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:     padding ParamUty    NA    NA     Inf              0 [NULL]
#> 3:      stride ParamUty    NA    NA     Inf         [NULL] [NULL]
#> 4:    dilation ParamUty    NA    NA     Inf              1 [NULL]
#> 5:   ceil_mode ParamLgl    NA    NA       2          FALSE [NULL]
```
