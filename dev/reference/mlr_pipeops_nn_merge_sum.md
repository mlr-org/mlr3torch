# Merge by Summation

Calculates the sum of all input tensors.

## nn_module

Calls
[`nn_merge_sum()`](https://mlr3torch.mlr-org.com/dev/reference/nn_merge_sum.md)
when trained.

## Parameters

No parameters.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

`PipeOpTorchMerge`s has either a *vararg* input channel if the
constructor argument `innum` is not set, or input channels `"input1"`,
..., `"input<innum>"`. There is one output channel `"output"`. For an
explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\>
[`PipeOpTorchMerge`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge.md)
-\> `PipeOpTorchMergeSum`

## Methods

### Public methods

- [`PipeOpTorchMergeSum$new()`](#method-PipeOpTorchMergeSum-initialize)

- [`PipeOpTorchMergeSum$clone()`](#method-PipeOpTorchMergeSum-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchMergeSum$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchMergeSum$new(id = "nn_merge_sum", innum = 0, param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `innum`:

  (`integer(1)`)  
  The number of inputs. Default is 0 which means there is one *vararg*
  input channel.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchMergeSum$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchMergeSum$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("merge_sum")
pipeop
#> 
#> ── PipeOp <merge_sum>: not trained ─────────────────────────────────────────────
#> Values: list()
#> 
#> ── Input channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>     ... ModelDescriptor    Task
#> 
#> ── Output channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>  output ModelDescriptor    Task
# The available parameters
pipeop$param_set
#> <ParamSet(0)>
#> Empty.
```
