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

Other PipeOps:
[`mlr_pipeops_nn_adaptive_avg_pool1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool1d.md),
[`mlr_pipeops_nn_adaptive_avg_pool2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool2d.md),
[`mlr_pipeops_nn_adaptive_avg_pool3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool3d.md),
[`mlr_pipeops_nn_avg_pool1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool1d.md),
[`mlr_pipeops_nn_avg_pool2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool2d.md),
[`mlr_pipeops_nn_avg_pool3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool3d.md),
[`mlr_pipeops_nn_batch_norm1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm1d.md),
[`mlr_pipeops_nn_batch_norm2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm2d.md),
[`mlr_pipeops_nn_batch_norm3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm3d.md),
[`mlr_pipeops_nn_block`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_block.md),
[`mlr_pipeops_nn_celu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_celu.md),
[`mlr_pipeops_nn_conv1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv1d.md),
[`mlr_pipeops_nn_conv2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv2d.md),
[`mlr_pipeops_nn_conv3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv3d.md),
[`mlr_pipeops_nn_conv_transpose1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose1d.md),
[`mlr_pipeops_nn_conv_transpose2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose2d.md),
[`mlr_pipeops_nn_conv_transpose3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose3d.md),
[`mlr_pipeops_nn_dropout`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_dropout.md),
[`mlr_pipeops_nn_elu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_elu.md),
[`mlr_pipeops_nn_flatten`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_flatten.md),
[`mlr_pipeops_nn_ft_cls`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_ft_cls.md),
[`mlr_pipeops_nn_ft_transformer_block`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_ft_transformer_block.md),
[`mlr_pipeops_nn_geglu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_geglu.md),
[`mlr_pipeops_nn_gelu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_gelu.md),
[`mlr_pipeops_nn_glu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_glu.md),
[`mlr_pipeops_nn_hardshrink`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardshrink.md),
[`mlr_pipeops_nn_hardsigmoid`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardsigmoid.md),
[`mlr_pipeops_nn_hardtanh`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardtanh.md),
[`mlr_pipeops_nn_head`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_head.md),
[`mlr_pipeops_nn_identity`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_identity.md),
[`mlr_pipeops_nn_layer_norm`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_layer_norm.md),
[`mlr_pipeops_nn_leaky_relu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_leaky_relu.md),
[`mlr_pipeops_nn_linear`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_linear.md),
[`mlr_pipeops_nn_log_sigmoid`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_log_sigmoid.md),
[`mlr_pipeops_nn_max_pool1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool1d.md),
[`mlr_pipeops_nn_max_pool2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool2d.md),
[`mlr_pipeops_nn_max_pool3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool3d.md),
[`mlr_pipeops_nn_merge`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge.md),
[`mlr_pipeops_nn_merge_cat`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_cat.md),
[`mlr_pipeops_nn_merge_prod`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_prod.md),
[`mlr_pipeops_nn_merge_sum`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_sum.md),
[`mlr_pipeops_nn_prelu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_prelu.md),
[`mlr_pipeops_nn_reglu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_reglu.md),
[`mlr_pipeops_nn_relu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_relu.md),
[`mlr_pipeops_nn_relu6`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_relu6.md),
[`mlr_pipeops_nn_reshape`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_reshape.md),
[`mlr_pipeops_nn_rrelu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_rrelu.md),
[`mlr_pipeops_nn_selu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_selu.md),
[`mlr_pipeops_nn_sigmoid`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_sigmoid.md),
[`mlr_pipeops_nn_softmax`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softmax.md),
[`mlr_pipeops_nn_softplus`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softplus.md),
[`mlr_pipeops_nn_softshrink`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softshrink.md),
[`mlr_pipeops_nn_softsign`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softsign.md),
[`mlr_pipeops_nn_squeeze`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_squeeze.md),
[`mlr_pipeops_nn_tanh`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tanh.md),
[`mlr_pipeops_nn_tanhshrink`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tanhshrink.md),
[`mlr_pipeops_nn_threshold`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_threshold.md),
[`mlr_pipeops_nn_tokenizer_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tokenizer_categ.md),
[`mlr_pipeops_nn_tokenizer_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tokenizer_num.md),
[`mlr_pipeops_nn_unsqueeze`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_unsqueeze.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model.md),
[`mlr_pipeops_torch_model_classif`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model_classif.md),
[`mlr_pipeops_torch_model_regr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model_regr.md)

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/reference/nn_graph.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3torch::PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md)
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
#> 1:      <tnsr[]>
#> 2:      <tnsr[]>
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
#>  -0.5130 -0.2689  0.9222  0.6628  0.7339  0.4802
#>   0.6101  0.4463 -0.6826 -0.9005 -1.9603  0.2701
#>  -0.3061 -0.2761 -0.4361 -0.3595  1.2065  0.5050
#>   0.9600 -1.0915 -0.0830 -0.3016  0.3335  0.5938
#>  -0.3497  1.1410 -0.4597  0.9940  0.5555  0.1082
#>   0.7665 -0.2785  0.0354 -0.5127  0.2003 -0.5144
#> 
#> (2,1,.,.) = 
#>   0.5583  1.0085  0.1419 -1.5090 -0.3322  0.0338
#>  -0.2030 -0.0904  0.1983 -0.3086  0.7198  0.6150
#>  -0.3464  0.1500  0.0761 -0.3187 -0.7971 -0.5778
#>   0.2333 -0.0740 -0.6963  0.5164  0.3149  0.6077
#>  -0.2064 -0.0729 -1.2075 -0.6682  0.1942 -0.4241
#>  -1.3678 -0.3263 -0.3250  1.1667 -0.5383  0.1821
#> 
#> (1,2,.,.) = 
#>  -0.1580  1.0942 -0.2101 -1.1956 -0.1334  0.5588
#>   0.3320 -0.9384 -0.5345  0.5592  0.2635 -0.1413
#>   0.1181 -0.8090 -0.0479  0.2616 -0.9342 -0.3589
#>  -0.2056 -0.9820 -0.9173 -0.6928 -0.1927 -1.1462
#>  -0.1020 -0.5876 -0.4189  0.4566 -1.7096 -0.2431
#>   1.1907 -0.0169 -0.9474 -0.7398 -0.3734  0.0747
#> 
#> (2,2,.,.) = 
#>  -1.3674  0.5566 -0.5345  0.5425  0.4601 -0.1522
#>   1.0683 -0.1674 -0.9388 -0.5893 -0.4362  0.2045
#>  -1.0404 -0.6271  0.5777 -0.2788  0.2541  0.5765
#>   0.4943  0.7681  1.4543 -0.7905 -0.6309 -0.6761
#>  -0.9870 -0.4970 -0.2400  0.1605 -0.2250  0.1134
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
