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
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)

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
#>  0.9198 -0.2892 -0.0248  0.1118 -0.5782 -0.1670
#>   0.5681  0.8767 -0.8574  0.2735 -0.2856  1.5497
#>   0.4787  0.1826  0.2233  0.6372 -1.2407  0.8789
#>  -0.2584 -0.5134  0.0906 -0.2702  1.4057 -0.3864
#>  -0.4198  1.5020 -0.6202 -0.0480 -1.0717  0.3076
#>  -0.1124 -0.5027  0.5909  0.0511  0.0618  0.2823
#> 
#> (2,1,.,.) = 
#>  0.3063 -0.7196  0.0712  0.8056  0.6160  1.2315
#>  -0.2108  0.0397  0.2036  0.4802 -0.0407  0.5115
#>  -0.4973  0.2581 -1.0943  1.4622  0.8741  0.0511
#>   0.2593 -0.4846 -0.0384 -0.9363  0.3541 -0.6121
#>   0.6645 -0.6691  0.4673  0.8148  0.3600 -1.0171
#>   2.0859 -0.3312 -0.7614 -0.6459 -0.8121  0.8095
#> 
#> (1,2,.,.) = 
#>  0.3283  0.8790  1.2608  0.8933  1.4261  1.4846
#>  -0.6632 -0.8042  0.3164  1.3274 -0.5722  0.6317
#>   0.0749  0.8481 -0.2305 -0.7817 -0.6154  0.7193
#>   0.9594 -0.8926  0.3845 -0.2179  0.7344 -0.2518
#>   0.1414 -0.4426 -0.3355 -0.6058  1.9757 -0.0658
#>  -0.7135 -1.0303  0.3557  0.2439  0.2089  0.0671
#> 
#> (2,2,.,.) = 
#> -1.1149 -0.3016  1.2582 -0.6004 -0.5959 -0.4440
#>   0.1747 -0.1078 -0.4480  0.6437 -0.2359  0.3160
#>  -0.3046  0.8761  0.9585  0.6702 -0.0370 -0.0495
#>  -1.1692 -1.1403  0.8120  0.4109 -1.8095  0.7932
#>  -0.6209  0.1325  0.2415  0.3656  0.4500 -0.0450
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
