# Output Head

Output head for classification and regresssion.

## Details

When the method `$shapes_out()` does not have access to the task, it
returns `c(NA, NA)`. When this
[`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html) is
trained however, the model descriptor has the correct output shape.

## nn_module

Calls
[`torch::nn_linear()`](https://torch.mlverse.org/docs/reference/nn_linear.html)
with the input features inferred from the input shape and the output
features from the task, via
[`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md).
For

- binary classification, the output dimension is 1.

- multiclass classification, the output dimension is the number of
  classes.

- regression, the output dimension is 1.

## Parameters

- `bias` :: `logical(1)`  
  Whether to use a bias. Default is `TRUE`.

## Supporting Other Task Types

The output dimension is not hard-coded here: `PipeOpTorchHead` asks the
generic
[`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md)
how many output neurons the task needs, and
[mlr3torch](https://CRAN.R-project.org/package=mlr3torch) implements
methods for
[`TaskClassif`](https://mlr3.mlr-org.com/reference/TaskClassif.html) and
[`TaskRegr`](https://mlr3.mlr-org.com/reference/TaskRegr.html). You can
add support to your custom task type by implementing a method for your
class.

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
-\> `PipeOpTorchHead`

## Methods

### Public methods

- [`PipeOpTorchHead$new()`](#method-PipeOpTorchHead-initialize)

- [`PipeOpTorchHead$clone()`](#method-PipeOpTorchHead-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchHead$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchHead$new(id = "nn_head", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchHead$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchHead$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("head")
pipeop
#> 
#> ── PipeOp <head>: not trained ──────────────────────────────────────────────────
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
#>        id    class lower upper nlevels default  value
#>    <char>   <char> <num> <num>   <num>  <list> <list>
#> 1:   bias ParamLgl    NA    NA       2    TRUE [NULL]
```
