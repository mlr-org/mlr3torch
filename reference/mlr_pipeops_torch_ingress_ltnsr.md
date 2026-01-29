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
#>   0.1497 -1.1231  0.7664 -0.1085  0.0787 -0.1360
#>  -0.7972  0.3721  0.0627  0.7924 -0.8084  0.3763
#>  -0.1213 -0.4353 -0.1649  0.0039 -0.7685  0.3077
#>  -0.7160 -0.4992 -0.3643 -0.5243  0.6776 -0.4790
#>  -0.0255 -1.3656 -1.0120 -0.5698 -0.2814 -0.8419
#>   0.9657 -0.1093  0.2461  0.4429 -0.7586 -0.7250
#> 
#> (2,1,.,.) = 
#>  -0.3701  0.2299 -1.5903 -0.4810  0.9107  0.2949
#>   1.1092 -0.3409 -0.6212  0.2951 -0.3930 -0.0518
#>  -0.4343  0.6685 -0.4598 -0.4484  0.0526  0.5092
#>   0.5382  0.7352  0.3505  0.2808 -0.3286 -0.4899
#>   0.1900  0.3942  0.7362 -0.1448  0.3445  0.6055
#>   0.9083 -0.1964 -0.1879 -0.0043  1.1188  0.3164
#> 
#> (1,2,.,.) = 
#>   0.1719 -0.1770 -0.8431  0.0950 -0.0132  0.4938
#>  -0.1290 -0.2649  0.5348  1.1194  0.6332  1.1016
#>   0.0745  0.0555 -0.3424 -0.0920  0.6357  0.6562
#>   0.3498 -0.3937  0.2380  0.3960 -1.2222 -0.2994
#>  -0.5304 -0.7835 -1.0263  0.4709 -0.7090 -1.4786
#>   0.0341 -0.1072  0.2773  0.1453 -0.0561  0.5105
#> 
#> (2,2,.,.) = 
#>   0.1458 -0.0208 -1.3172 -0.8632  0.4697 -0.2444
#>   0.6186  0.5026  0.8383  0.4956  0.0317  0.0670
#>  -0.1436  0.1410 -0.8852  1.5472 -1.1013 -0.5768
#>   0.3191  0.6680 -1.4505 -0.1896  0.5049 -0.1750
#>   0.4367  0.1651  0.5050 -0.0582 -0.1118  0.7806
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{2,3,6,6} ]
```
