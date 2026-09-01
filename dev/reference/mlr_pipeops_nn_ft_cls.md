# CLS Token for FT-Transformer

Concatenates a CLS token to the input as the last feature. The input
shape is expected to be `(batch, n_features, d_token)` and the output
shape is `(batch, n_features + 1, d_token)`.

This is used in the
[`LearnerTorchFTTransformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md).

## nn_module

Calls
[`nn_ft_cls()`](https://mlr3torch.mlr-org.com/dev/reference/nn_ft_cls.md)
when trained.

## Parameters

|                |           |         |                 |
|----------------|-----------|---------|-----------------|
| Id             | Type      | Default | Levels          |
| initialization | character | \-      | uniform, normal |

## State

The state is the value calculated by the public method `$shapes_out()`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchFTCLS`

## Methods

### Public methods

- [`PipeOpTorchFTCLS$new()`](#method-PipeOpTorchFTCLS-initialize)

- [`PipeOpTorchFTCLS$clone()`](#method-PipeOpTorchFTCLS-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchFTCLS$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchFTCLS$new(id = "nn_ft_cls", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchFTCLS$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchFTCLS$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("ft_cls")
pipeop
#> 
#> ── PipeOp <ft_cls>: not trained ────────────────────────────────────────────────
#> Values: initialization=uniform
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
#>                id    class lower upper nlevels        default   value
#>            <char>   <char> <num> <num>   <num>         <list>  <list>
#> 1: initialization ParamFct    NA    NA       2 <NoDefault[0]> uniform
```
