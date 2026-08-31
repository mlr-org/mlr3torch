# Ingress for Lazy Tensor

Ingress for a single
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
column.

## Parameters

- `shape` :: [`integer()`](https://rdrr.io/r/base/integer.html) \|
  `NULL` \| `"infer"`  
  The shape of the tensor, where the first dimension (batch) must be
  `NA`. When it is not specified, the lazy tensor input column needs to
  have a known shape. When it is set to `"infer"`, the shape is inferred
  from an example batch.

## Internals

The returned batchgetter materializes the lazy tensor column to a
tensor.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is set to the input shape.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md),
[`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md)
-\> `PipeOpTorchIngressLazyTensor`

## Methods

### Public methods

- [`PipeOpTorchIngressLazyTensor$new()`](#method-PipeOpTorchIngressLazyTensor-initialize)

- [`PipeOpTorchIngressLazyTensor$clone()`](#method-PipeOpTorchIngressLazyTensor-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchIngressLazyTensor$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchIngressLazyTensor$new(
      id = "torch_ingress_ltnsr",
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchIngressLazyTensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchIngressLazyTensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
po_ingress = po("torch_ingress_ltnsr")
task = tsk("lazy_iris")

md = po_ingress$train(list(task))[[1L]]

ingress = md$ingress
x_batch = ingress[[1L]]$batchgetter(data = task$data(1, "x"), cache = NULL)
x_batch
#> torch_tensor
#>  5.1000  3.5000  1.4000  0.2000
#> [ CPUFloatType{1,4} ]

# Now we try a lazy tensor with unknown shape, i.e. the shapes between the rows can differ

ds = dataset(
  initialize = function() self$x = list(torch_randn(3, 10, 10), torch_randn(3, 8, 8)),
  .getitem = function(i) list(x = self$x[[i]]),
  .length = function() 2)()

task_unknown = as_task_regr(data.table(
  x = as_lazy_tensor(ds, dataset_shapes = list(x = NULL)),
  y = rnorm(2)
), target = "y", id = "example2")

# this task (as it is) can NOT be processed by PipeOpTorchIngressLazyTensor
# It therefore needs to be preprocessed
po_resize = po("trafo_resize", size = c(6, 6))
task_unknown_resize = po_resize$train(list(task_unknown))[[1L]]

# printing the transformed column still shows unknown shapes,
# because the preprocessing pipeop cannot infer them,
# however we know that the shape is now (3, 10, 10) for all rows
task_unknown_resize$data(1:2, "x")
#>                x
#>    <lazy_tensor>
#> 1:     <tnsr[?]>
#> 2:     <tnsr[?]>
po_ingress$param_set$set_values(shape = c(NA, 3, 6, 6))

md2 = po_ingress$train(list(task_unknown_resize))[[1L]]

ingress2 = md2$ingress
x_batch2 = ingress2[[1L]]$batchgetter(
  data = task_unknown_resize$data(1:2, "x"),
  cache = NULL
)

x_batch2
#> torch_tensor
#> (1,1,.,.) = 
#> -0.1727 -1.1561 -0.8809 -0.1884 -0.1366  0.5646
#>   0.6274 -0.3574 -1.3656 -0.3476 -0.4548 -0.3863
#>   0.6161  0.0090 -0.2375  0.1474  0.6387 -0.2104
#>  -1.1592 -0.5147 -0.8126  0.4269 -0.6004 -0.4679
#>  -1.0021  0.4385 -1.6420 -0.8544 -0.7600  0.3681
#>  -0.5659  0.2289 -0.0646  0.4244 -0.3786  0.2266
#> 
#> (2,1,.,.) = 
#> -0.4253 -1.0378 -0.4453 -1.3907  0.2681 -1.1969
#>  -0.7326  0.0546 -0.8478  0.1833  0.0585  0.0901
#>   0.3821  0.5556  0.0447  0.2189 -0.5316  0.2044
#>   1.5000  0.4006 -0.0095  0.1972 -0.2539  0.4295
#>   1.0999  0.6315  0.2035  0.2077  0.5334  0.3706
#>   1.1029  0.3964 -0.2185 -0.8719 -0.6402  1.0154
#> 
#> (1,2,.,.) = 
#> -0.3867 -0.1485  0.7578  0.1674 -0.8712 -0.6235
#>  -0.1103 -0.0947 -0.3690 -0.1131  0.6638 -0.6330
#>  -0.2525 -0.0587 -1.1791 -0.0575 -0.6642  0.5305
#>   0.1051 -1.1957  0.0863  0.5499  0.5872  0.5805
#>   2.1126 -1.1811 -0.3699 -0.2372 -1.4690  0.0022
#>  -0.2731 -0.5973  0.3913  0.1769  0.0431 -0.5305
#> 
#> (2,2,.,.) = 
#>  0.8543 -0.1859  0.2522 -0.1430  0.0387  0.9980
#>   0.2235  0.4589  0.1187 -1.3088 -0.2525  0.1584
#>  -0.2568 -0.5714  0.0538 -0.6353  0.3687  0.9067
#>  -0.6447 -1.1811 -0.4068 -0.1484 -0.4457 -1.1012
#>  -0.3940 -0.5193 -0.4492  0.1964 -0.1741  0.0056
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
