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
[`mlr_pipeops_nn_multihead_attention`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_multihead_attention.md),
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
#>  0.9485  0.3380  0.0509 -0.2982  0.1846  0.3651
#>   0.3373 -1.3674  0.0049 -1.1276 -0.7581  0.6527
#>   0.2916 -0.5591 -0.0207  0.1179  0.2934 -0.0059
#>  -0.4277  0.1694  0.3361 -0.3255 -0.1427 -0.2486
#>  -0.2362 -0.3481 -0.5518 -0.5082 -1.0303 -0.3004
#>   0.6732  0.0705 -0.0956  0.3198 -1.9436  0.7129
#> 
#> (2,1,.,.) = 
#>  0.5675  0.5569 -0.3859 -0.1925  0.3140 -0.3008
#>  -0.0872  1.5720  1.3287  0.3445  0.2794 -0.6375
#>  -0.1817  0.4953  0.8520 -1.7477 -0.2826 -0.8929
#>  -0.5015  0.1065 -0.4863 -0.1249 -0.0318 -0.7105
#>  -0.5277  0.4021 -0.3630  0.5837  0.0869  0.6156
#>   0.0847  0.7729 -0.7570 -0.4195  1.2477  0.6399
#> 
#> (1,2,.,.) = 
#>  0.6718  0.1022  0.2428 -0.6244  0.2885 -0.5892
#>  -0.6850  1.1433  0.0798 -0.6273  0.1845  0.3467
#>  -0.2617  0.1084  0.1700 -0.4283 -0.1421 -0.1030
#>   0.5947 -0.8087 -0.2139 -1.2183  0.8797 -0.2916
#>  -0.3419  1.0483 -0.5350 -0.2892  1.5265  0.1381
#>  -0.3289  0.3515  0.4301 -0.7378 -0.9259 -0.4091
#> 
#> (2,2,.,.) = 
#> -0.3355  0.0284 -0.4499 -2.1221 -0.4670  1.1491
#>  -0.2810  0.2817  0.1438  0.0380 -0.7249 -1.1667
#>   1.2589  0.0791 -0.3634  0.1357  1.0075  2.2482
#>  -0.4658  0.4100 -0.2441 -0.6835  0.3243 -0.1052
#>   0.5225  0.1970 -0.7998  0.0694 -0.4536 -0.0276
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
