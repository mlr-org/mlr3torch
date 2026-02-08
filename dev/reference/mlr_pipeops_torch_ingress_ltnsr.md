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

Other PipeOps:
[`mlr_pipeops_nn_adaptive_avg_pool1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_adaptive_avg_pool1d.md),
[`mlr_pipeops_nn_adaptive_avg_pool2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_adaptive_avg_pool2d.md),
[`mlr_pipeops_nn_adaptive_avg_pool3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_adaptive_avg_pool3d.md),
[`mlr_pipeops_nn_avg_pool1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_avg_pool1d.md),
[`mlr_pipeops_nn_avg_pool2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_avg_pool2d.md),
[`mlr_pipeops_nn_avg_pool3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_avg_pool3d.md),
[`mlr_pipeops_nn_batch_norm1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_batch_norm1d.md),
[`mlr_pipeops_nn_batch_norm2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_batch_norm2d.md),
[`mlr_pipeops_nn_batch_norm3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_batch_norm3d.md),
[`mlr_pipeops_nn_block`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_block.md),
[`mlr_pipeops_nn_celu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_celu.md),
[`mlr_pipeops_nn_conv1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv1d.md),
[`mlr_pipeops_nn_conv2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv2d.md),
[`mlr_pipeops_nn_conv3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv3d.md),
[`mlr_pipeops_nn_conv_transpose1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv_transpose1d.md),
[`mlr_pipeops_nn_conv_transpose2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv_transpose2d.md),
[`mlr_pipeops_nn_conv_transpose3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_conv_transpose3d.md),
[`mlr_pipeops_nn_dropout`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_dropout.md),
[`mlr_pipeops_nn_elu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_elu.md),
[`mlr_pipeops_nn_flatten`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_flatten.md),
[`mlr_pipeops_nn_ft_cls`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_ft_cls.md),
[`mlr_pipeops_nn_ft_transformer_block`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_ft_transformer_block.md),
[`mlr_pipeops_nn_geglu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_geglu.md),
[`mlr_pipeops_nn_gelu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_gelu.md),
[`mlr_pipeops_nn_glu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_glu.md),
[`mlr_pipeops_nn_hardshrink`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_hardshrink.md),
[`mlr_pipeops_nn_hardsigmoid`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_hardsigmoid.md),
[`mlr_pipeops_nn_hardtanh`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_hardtanh.md),
[`mlr_pipeops_nn_head`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md),
[`mlr_pipeops_nn_identity`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_identity.md),
[`mlr_pipeops_nn_layer_norm`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_layer_norm.md),
[`mlr_pipeops_nn_leaky_relu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_leaky_relu.md),
[`mlr_pipeops_nn_linear`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_linear.md),
[`mlr_pipeops_nn_log_sigmoid`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_log_sigmoid.md),
[`mlr_pipeops_nn_max_pool1d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_max_pool1d.md),
[`mlr_pipeops_nn_max_pool2d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_max_pool2d.md),
[`mlr_pipeops_nn_max_pool3d`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_max_pool3d.md),
[`mlr_pipeops_nn_merge`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge.md),
[`mlr_pipeops_nn_merge_cat`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_cat.md),
[`mlr_pipeops_nn_merge_prod`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_prod.md),
[`mlr_pipeops_nn_merge_sum`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_sum.md),
[`mlr_pipeops_nn_prelu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_prelu.md),
[`mlr_pipeops_nn_reglu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_reglu.md),
[`mlr_pipeops_nn_relu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_relu.md),
[`mlr_pipeops_nn_relu6`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_relu6.md),
[`mlr_pipeops_nn_reshape`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_reshape.md),
[`mlr_pipeops_nn_rrelu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_rrelu.md),
[`mlr_pipeops_nn_selu`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_selu.md),
[`mlr_pipeops_nn_sigmoid`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_sigmoid.md),
[`mlr_pipeops_nn_softmax`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softmax.md),
[`mlr_pipeops_nn_softplus`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softplus.md),
[`mlr_pipeops_nn_softshrink`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softshrink.md),
[`mlr_pipeops_nn_softsign`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_softsign.md),
[`mlr_pipeops_nn_squeeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_squeeze.md),
[`mlr_pipeops_nn_tanh`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tanh.md),
[`mlr_pipeops_nn_tanhshrink`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tanhshrink.md),
[`mlr_pipeops_nn_threshold`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_threshold.md),
[`mlr_pipeops_nn_tokenizer_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tokenizer_categ.md),
[`mlr_pipeops_nn_tokenizer_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_tokenizer_num.md),
[`mlr_pipeops_nn_unsqueeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_unsqueeze.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model.md),
[`mlr_pipeops_torch_model_classif`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model_classif.md),
[`mlr_pipeops_torch_model_regr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model_regr.md)

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
[`mlr3torch::PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md)
-\> `PipeOpTorchIngressLazyTensor`

## Methods

### Public methods

- [`PipeOpTorchIngressLazyTensor$new()`](#method-PipeOpTorchIngressLazyTensor-new)

- [`PipeOpTorchIngressLazyTensor$clone()`](#method-PipeOpTorchIngressLazyTensor-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### Method `new()`

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

### Method `clone()`

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
#> 1:     <list[2]>
#> 2:     <list[2]>
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
#>  -0.3608  0.5216  0.1733  0.6437  0.1040  0.1914
#>   0.3499  0.8800 -0.0178  0.0569 -2.4122 -0.5805
#>  -0.6747  0.3290  0.0024 -0.0736  1.2566  0.6037
#>   0.4681 -1.0867 -0.1721 -0.0527  0.9650  0.7552
#>  -0.3569 -1.6199 -0.2174 -1.2658  1.4037  1.2540
#>   0.0444  0.6583  1.0370 -0.3837 -0.8354  0.1146
#> 
#> (2,1,.,.) = 
#>   0.0899 -0.3703 -0.9690 -0.8399 -0.4300 -0.3315
#>  -0.1023  0.0553 -0.8862 -1.2003  0.1211  0.5432
#>  -1.1030 -0.0422 -0.8181 -0.6470 -1.0169  1.4592
#>   0.5326  0.9041  0.3137  0.1626 -0.3241 -0.0354
#>   0.7098  0.1096 -0.1567  0.2802 -0.0381 -0.1842
#>   0.0851  0.9778  0.3168 -0.5295 -0.6983  0.2748
#> 
#> (1,2,.,.) = 
#>   0.6109 -0.2358 -0.5721  0.8212  0.8442 -0.3173
#>  -2.0135 -0.6595 -0.9571  0.1580  1.7528  0.0527
#>  -0.0905  1.2738  0.3565 -0.8008 -1.1047  0.1361
#>   1.1674  0.4496 -0.2545  0.5400  0.5283 -0.5629
#>   1.0017 -1.0180 -0.0231 -0.9276  0.1300  0.6725
#>  -1.0019  1.8710  0.0371 -1.0671 -0.5614  0.3651
#> 
#> (2,2,.,.) = 
#>  -0.5957  0.0657 -0.4566 -0.4003  0.0872 -0.6612
#>   0.0052  0.0215 -0.8740 -0.4637 -1.2063 -0.8745
#>   1.1247  0.2833 -1.2461 -0.3965  0.2146  0.5675
#>  -1.5753  0.3289  0.3668  0.1033 -0.4556 -0.3684
#>  -0.4959  0.3279 -0.4048  0.0789  0.3688  0.1760
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
