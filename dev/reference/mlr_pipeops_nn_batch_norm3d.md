# 3D Batch Normalization

Applies Batch Normalization for each channel across a batch of data.

## nn_module

Calls
[`torch::nn_batch_norm3d()`](https://torch.mlverse.org/docs/reference/nn_batch_norm3d.html).
The parameter `num_features` is inferred as the second dimension of the
input shape.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Parameters

- `eps` :: `numeric(1)`  
  A value added to the denominator for numerical stability. Default:
  `1e-5`.

- `momentum` :: `numeric(1)`  
  The value used for the running_mean and running_var computation. Can
  be set to `NULL` for cumulative moving average (i.e. simple average).
  Default: 0.1

- `affine` :: `logical(1)`  
  a boolean value that when set to `TRUE`, this module has learnable
  affine parameters. Default: `TRUE`

- `track_running_stats` :: `logical(1)`  
  a boolean value that when set to `TRUE`, this module tracks the
  running mean and variance, and when set to `FALSE`, this module does
  not track such statistics and always uses batch statistics in both
  training and eval modes. Default: `TRUE`

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchBatchNorm` -\> `PipeOpTorchBatchNorm3D`

## Methods

### Public methods

- [`PipeOpTorchBatchNorm3D$new()`](#method-PipeOpTorchBatchNorm3D-initialize)

- [`PipeOpTorchBatchNorm3D$clone()`](#method-PipeOpTorchBatchNorm3D-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchBatchNorm3D$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchBatchNorm3D$new(id = "nn_batch_norm3d", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchBatchNorm3D$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchBatchNorm3D$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("batch_norm3d")
pipeop
#> 
#> ── PipeOp <batch_norm3d>: not trained ──────────────────────────────────────────
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
#> <ParamSet(4)>
#>                     id    class lower upper nlevels default  value
#>                 <char>   <char> <num> <num>   <num>  <list> <list>
#> 1:                 eps ParamDbl     0   Inf     Inf   1e-05 [NULL]
#> 2:            momentum ParamDbl     0   Inf     Inf     0.1 [NULL]
#> 3:              affine ParamLgl    NA    NA       2    TRUE [NULL]
#> 4: track_running_stats ParamLgl    NA    NA       2    TRUE [NULL]
```
