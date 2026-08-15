# SELU Activation Function

Applies element-wise, \$\$SELU(x) = scale \* (max(0,x) + min(0, \alpha
\* (exp(x) - 1)))\$\$, with \\\alpha=1.6732632423543772848170429916717\\
and \\scale=1.0507009873554804934193349852946\\.

## nn_module

Calls
[`torch::nn_selu()`](https://torch.mlverse.org/docs/reference/nn_selu.html)
when trained.

## Parameters

- `inplace` :: `logical(1)`  
  Whether to do the operation in-place. Default: `FALSE`.

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
-\> `PipeOpTorchSELU`

## Methods

### Public methods

- [`PipeOpTorchSELU$new()`](#method-PipeOpTorchSELU-initialize)

- [`PipeOpTorchSELU$clone()`](#method-PipeOpTorchSELU-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchSELU$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchSELU$new(id = "nn_selu", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchSELU$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchSELU$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("selu")
pipeop
#> 
#> ── PipeOp <selu>: not trained ──────────────────────────────────────────────────
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
#>         id    class lower upper nlevels default  value
#>     <char>   <char> <num> <num>   <num>  <list> <list>
#> 1: inplace ParamLgl    NA    NA       2   FALSE [NULL]
```
