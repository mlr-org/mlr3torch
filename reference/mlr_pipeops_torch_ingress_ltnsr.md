# Ingress for Lazy Tensor

Ingress for a single
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md).

## State

The state is set to the input shape.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md),
[`as_learner_torch()`](https://mlr3torch.mlr-org.com/reference/as_learner_torch.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/reference/nn_graph.md),
[`pipeop_torch()`](https://mlr3torch.mlr-org.com/reference/pipeop_torch.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md)
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
#>  0.6941  0.3028  0.2809 -0.2410 -0.2744  0.3191
#>  -0.7846 -0.9853 -1.6607 -0.2194  1.5062 -0.1494
#>  -0.9507  0.7458 -0.1833  0.6345 -0.1759 -0.3677
#>   0.2974 -1.7051  1.2708  0.0448 -0.7055 -0.6369
#>   0.4819  0.2980  0.2857 -0.4478 -0.8208  0.4438
#>  -0.1120 -0.2456  0.4549 -0.1352  0.0786 -0.8345
#> 
#> (2,1,.,.) = 
#>  0.0473 -0.3885  0.0879 -0.7678  0.0023  1.0875
#>   0.0094  0.3401 -0.0924 -0.7253 -0.0081  0.4713
#>   0.8276 -0.7216 -0.2193 -0.7811  0.9276  0.4806
#>   0.3485  0.1451  0.1984  0.1896  0.6990  0.7853
#>   1.2705 -0.1965 -0.3804  0.2149  0.6428  0.2717
#>   1.6391  0.2042  1.8030 -0.9482 -0.1900  0.1831
#> 
#> (1,2,.,.) = 
#>  0.4976  0.6018 -1.1406 -0.7116 -0.3748  0.0641
#>  -1.0526 -0.1047 -0.2903  0.1551 -1.1351 -0.1248
#>  -0.5064  1.0382  0.5404  0.3237  1.0622  0.1847
#>   0.0049  0.9869  0.7275  0.1360 -0.7176  0.0864
#>   0.3871 -0.2940 -0.0906 -1.3450  2.0662  0.5440
#>   0.7734  0.1991  0.5636 -0.2431 -0.9927 -0.0338
#> 
#> (2,2,.,.) = 
#>  0.8883  0.3728  0.7696  0.0602 -0.1931  1.7063
#>   0.4424  0.2059  0.7263 -0.0463  0.4192  0.1241
#>   0.0367  0.6619  0.1231 -0.3101 -0.1519 -0.3070
#>   0.2171 -0.3469 -1.1674  0.0939  0.2133  0.5434
#>   0.0773  0.3440 -0.2846 -0.4997  0.5567  0.0098
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
