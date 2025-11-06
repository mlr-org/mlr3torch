# Custom Function

Applies a user-supplied function to a tensor.

## Parameters

By default, these are inferred as all but the first arguments of the
function `fn`. It is also possible to specify these more explicitly via
the `param_set` constructor argument.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md).

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3torch::PipeOpTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchFn`

## Methods

### Public methods

- [`PipeOpTorchFn$new()`](#method-PipeOpTorchFn-new)

- [`PipeOpTorchFn$clone()`](#method-PipeOpTorchFn-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`mlr3torch::PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[`R6`](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchFn$new(
      fn,
      id = "nn_fn",
      param_vals = list(),
      param_set = NULL,
      shapes_out = NULL
    )

#### Arguments

- `fn`:

  (`function`)  
  The function to be applied. Takes a `torch` tensor as first argument
  and returns a `torch` tensor.

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) or
  `NULL`)  
  A ParamSet wrapping the arguments to `fn`. If omitted, then the
  ParamSet for this PipeOp will be inferred from the function signature.

- `shapes_out`:

  (`function` or `NULL`)  
  A function that computes the output shapes of the `fn`. See
  [PipeOpTorch](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md)'s
  `.shapes_out()` method for details on the parameters, and
  [PipeOpTaskPreprocTorch](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_preproc_torch.md)
  for details on how the shapes are inferred when this parameter is
  `NULL`.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchFn$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
custom_fn =  function(x, a) x / a
obj = po("nn_fn", fn = custom_fn, a = 2)
obj$param_set
#> <ParamSet(1)>
#>        id    class lower upper nlevels        default  value
#>    <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:      a ParamUty    NA    NA     Inf <NoDefault[0]>      2

graph = po("torch_ingress_ltnsr") %>>% obj

task = tsk("lazy_iris")$filter(1)
tnsr = materialize(task$data()$x)[[1]]

md_trained = graph$train(task)
trained = md_trained[[1]]$graph$train(tnsr)

trained[[1]]
#> torch_tensor
#>  2.5500
#>  1.7500
#>  0.7000
#>  0.1000
#> [ CPUFloatType{4} ]

custom_fn(tnsr, a = 2)
#> torch_tensor
#>  2.5500
#>  1.7500
#>  0.7000
#>  0.1000
#> [ CPUFloatType{4} ]
```
