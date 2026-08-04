# Multi-Head Attention

Multi-head attention as described in *Attention Is All You Need*.

This is a thin wrapper around
[`torch::nn_multihead_attention()`](https://torch.mlverse.org/docs/reference/nn_multihead_attention.html)
that makes it usable as a building block of a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) of
tensor operations, where both self-attention and cross-attention can be
expressed, see section *Input and Output Channels*.

## Tensor Layout

All inputs and outputs are `(batch, sequence, feature)`, i.e. the
`batch_first` layout of
[`torch::nn_multihead_attention()`](https://torch.mlverse.org/docs/reference/nn_multihead_attention.html),
which is fixed and not a hyperparameter. `torch` defaults to
`(sequence, batch, feature)`, but the first dimension of every shape has
to be the batch dimension here.

## nn_module

Calls
[`torch::nn_multihead_attention()`](https://torch.mlverse.org/docs/reference/nn_multihead_attention.html)
when trained, where the parameters `embed_dim`, `kdim` and `vdim` are
inferred as the last dimension of the query, key and value tensors
respectively, and `batch_first` is always `TRUE`, see section *Tensor
Layout*.

## Parameters

- `num_heads` :: `integer(1)`  
  Number of parallel attention heads. The embedding dimension must be
  divisible by `num_heads`.

- `dropout` :: `numeric(1)`  
  Dropout probability on the attention weights. Default is `0`.

- `bias` :: `logical(1)`  
  Whether to add a bias to the input and output projections. Default is
  `TRUE`.

- `add_bias_kv` :: `logical(1)`  
  Whether to add a bias to the key and value sequences at dimension 1.
  Default is `FALSE`.

- `add_zero_attn` :: `logical(1)`  
  Whether to add a new batch of zeros to the key and value sequences at
  dimension 1. Default is `FALSE`.

- `avg_weights` :: `logical(1)`  
  Whether the returned attention weights are averaged over the attention
  heads. Default is `TRUE`. Only has an effect when the construction
  argument `need_weights` is `TRUE`.

Note that `embed_dim`, `kdim` and `vdim` are *not* parameters, as they
are inferred from the shapes of the input tensors, and that
`batch_first` is *not* a parameter either, as it is fixed to `TRUE`, see
section *Tensor Layout*.

## Input and Output Channels

The number of input channels is determined by the construction argument
`mode`:

- `mode = "self"` (default): one input channel `"input"`, which is used
  as query, key and value, i.e. the `PipeOp` performs *self-attention*.

- `mode = "cross"`: input channels `"query"` and `"key_value"`, i.e. the
  `PipeOp` performs *cross-attention*, where the second input is used as
  both key and value.

- `mode = "general"`: input channels `"query"`, `"key"` and `"value"`,
  i.e. the `PipeOp` performs *cross-attention* with separate key and
  value inputs.

The number of output channels is determined by the construction argument
`need_weights`:

- `need_weights = FALSE` (default): one output channel `"output"`,
  containing the attention output.

- `need_weights = TRUE`: output channels `"output"` and `"weights"`,
  where the latter contains the attention weights.

For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## References

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez A, Kaiser Ł,
Polosukhin I (2017). “Attention is all you need.” *Advances in neural
information processing systems*, **30**.

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
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model.md),
[`mlr_pipeops_torch_model_classif`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model_classif.md),
[`mlr_pipeops_torch_model_regr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model_regr.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchMultiheadAttention`

## Methods

### Public methods

- [`PipeOpTorchMultiheadAttention$new()`](#method-PipeOpTorchMultiheadAttention-initialize)

- [`PipeOpTorchMultiheadAttention$clone()`](#method-PipeOpTorchMultiheadAttention-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchMultiheadAttention$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchMultiheadAttention$new(
      id = "nn_multihead_attention",
      mode = "self",
      need_weights = FALSE,
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `mode`:

  (`character(1)`)  
  The attention mode, which determines the input channels. One of
  `"self"`, `"cross"` or `"general"`. This is a *construction* argument
  (and not a hyperparameter), because it determines the structure of the
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html). The
  default is `"self"`, which means that the `PipeOp` performs
  self-attention. See section *Input and Output Channels* for more
  information.

- `need_weights`:

  (`logical(1)`)  
  Whether the attention weights are returned in addition to the
  attention output, i.e. whether there is a second output channel
  `"weights"`. This is a *construction* argument (and not a
  hyperparameter), because it determines the structure of the
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html). The
  default is `FALSE`, which means that only the attention output is
  returned. See section *Input and Output Channels* for more
  information.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchMultiheadAttention$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchMultiheadAttention$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = po("nn_multihead_attention", num_heads = 4)
pipeop
#> 
#> ── PipeOp <nn_multihead_attention>: not trained ────────────────────────────────
#> Values: num_heads=4
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
#> <ParamSet(6)>
#>               id    class lower upper nlevels        default  value
#>           <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:     num_heads ParamInt     1   Inf     Inf <NoDefault[0]>      4
#> 2:       dropout ParamDbl     0     1     Inf              0 [NULL]
#> 3:          bias ParamLgl    NA    NA       2           TRUE [NULL]
#> 4:   add_bias_kv ParamLgl    NA    NA       2          FALSE [NULL]
#> 5: add_zero_attn ParamLgl    NA    NA       2          FALSE [NULL]
#> 6:   avg_weights ParamLgl    NA    NA       2           TRUE [NULL]
```
