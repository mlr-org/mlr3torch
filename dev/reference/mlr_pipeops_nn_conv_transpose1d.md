# Transpose 1D Convolution

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

## nn_module

Calls
[`nn_conv_transpose1d`](https://torch.mlverse.org/docs/reference/nn_conv_transpose1d.html).
The parameter `in_channels` is inferred as the second dimension of the
input tensor.

## Parameters

- `out_channels` :: `integer(1)`  
  Number of output channels produce by the convolution.

- `kernel_size` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Size of the convolving kernel.

- `stride` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Stride of the convolution. Default: 1.

- `padding` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of the input. Default: 0.

- `output_padding` ::
  [`integer()`](https://rdrr.io/r/base/integer.html)  
  Additional size added to one side of the output shape. Default: 0.

- `groups` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Number of blocked connections from input channels to output channels.
  Default: 1

- `bias` :: `logical(1)`  
  If `True`, adds a learnable bias to the output. Default: `TRUE`.

- `dilation` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  Spacing between kernel elements. Default: 1.

- `padding_mode` :: `character(1)`  
  The padding mode. One of `"zeros"`, `"reflect"`, `"replicate"`, or
  `"circular"`. Default is `"zeros"`.

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
-\> `PipeOpTorchConvTranspose` -\> `PipeOpTorchConvTranspose1D`

## Methods

### Public methods

- [`PipeOpTorchConvTranspose1D$new()`](#method-PipeOpTorchConvTranspose1D-initialize)

- [`PipeOpTorchConvTranspose1D$clone()`](#method-PipeOpTorchConvTranspose1D-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchConvTranspose1D$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchConvTranspose1D$new(id = "nn_conv_transpose1d", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchConvTranspose1D$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchConvTranspose1D$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("conv_transpose1d", kernel_size = 3, out_channels = 2)
pipeop
#> 
#> ── PipeOp <conv_transpose1d>: not trained ──────────────────────────────────────
#> Values: out_channels=2, kernel_size=3
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
#> <ParamSet(9)>
#>                id    class lower upper nlevels        default  value
#>            <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:   out_channels ParamInt     1   Inf     Inf <NoDefault[0]>      2
#> 2:    kernel_size ParamUty    NA    NA     Inf <NoDefault[0]>      3
#> 3:         stride ParamUty    NA    NA     Inf              1 [NULL]
#> 4:        padding ParamUty    NA    NA     Inf              0 [NULL]
#> 5: output_padding ParamUty    NA    NA     Inf              0 [NULL]
#> 6:       dilation ParamUty    NA    NA     Inf              1 [NULL]
#> 7:         groups ParamInt     1   Inf     Inf              1 [NULL]
#> 8:           bias ParamLgl    NA    NA       2           TRUE [NULL]
#> 9:   padding_mode ParamFct    NA    NA       4          zeros [NULL]
```
