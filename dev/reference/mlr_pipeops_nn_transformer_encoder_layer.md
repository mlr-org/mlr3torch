# Transformer Encoder Layer

A single transformer encoder layer as described in *Attention Is All You
Need*, consisting of a multi-head self-attention block and a
position-wise feed-forward network, each wrapped in a residual
connection and layer normalization.

This is a thin wrapper around
[`torch::nn_transformer_encoder_layer()`](https://torch.mlverse.org/docs/reference/nn_transformer_encoder_layer.html)
that makes it usable as a building block of a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) of
tensor operations.

## Tensor Layout

Input and output are `(batch, sequence, feature)`, i.e. the
`batch_first` layout of
[`torch::nn_transformer_encoder_layer()`](https://torch.mlverse.org/docs/reference/nn_transformer_encoder_layer.html),
which is fixed and not a hyperparameter. `torch` defaults to
`(sequence, batch, feature)`, but the first dimension of every shape has
to be the batch dimension here.

The layer preserves the shape of its input, so the output shape is the
input shape. Only the feature dimension has to be known, the sequence
length may be `NA`.

## nn_module

Calls
[`torch::nn_transformer_encoder_layer()`](https://torch.mlverse.org/docs/reference/nn_transformer_encoder_layer.html)
when trained, where the parameter `d_model` is inferred as the last
dimension of the input tensor and `batch_first` is always `TRUE`, see
section *Tensor Layout*.

## Parameters

- `nhead` :: `integer(1)`  
  Number of parallel attention heads. The feature dimension must be
  divisible by `nhead`.

- `dim_feedforward` :: `integer(1)`  
  The hidden dimension of the feed-forward network. Default is `2048`.

- `dropout` :: `numeric(1)`  
  Dropout probability, applied to both the attention weights and the
  feed-forward network. Default is `0.1`.

- `activation` :: `character(1)` \| `function`  
  The activation function of the feed-forward network, either `"relu"`,
  `"gelu"`, or a function mapping a
  [`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  to a
  [`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html).
  Default is `"relu"`.

- `layer_norm_eps` :: `numeric(1)`  
  A value added to the denominator of the layer normalizations for
  numerical stability. Default is `1e-5`.

- `norm_first` :: `logical(1)`  
  Whether to apply layer normalization before the attention and
  feed-forward sublayers (pre-norm) instead of after them (post-norm).
  Default is `FALSE`.

- `bias` :: `logical(1)`  
  Whether the linear layers use a bias and the layer normalizations
  learn affine parameters. Default is `TRUE`.

- `is_causal` :: `logical(1)`  
  Whether to apply a causal mask, i.e. whether a position may only
  attend to the positions up to and including itself. This cannot be
  combined with the `"src_mask"` input channel. Default is `FALSE`.

Note that `d_model` is *not* a parameter, as it is inferred from the
shape of the input tensor, and that `batch_first` is *not* a parameter
either, as it is fixed to `TRUE`, see section *Tensor Layout*.

## Input and Output Channels

There is always one input channel `"input"`, which is the sequence the
layer attends over, and one output channel `"output"` of the same shape.

The construction arguments `src_mask` and `src_key_padding_mask` each
add a further input channel of that name, which is passed to the
corresponding argument of the module's `forward()` method. They are
construction arguments and not hyperparameters, because they determine
the structure of the
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html):

- `"src_mask"`: an additive or boolean mask over the sequence, of shape
  `(sequence, sequence)` or `(batch * nhead, sequence, sequence)`.
  Positions that are `TRUE` are *not* attended to.

- `"src_key_padding_mask"`: a mask of shape `(batch, sequence)` marking
  the padding positions of each observation, which are `TRUE`.

Note that a `(sequence, sequence)` `"src_mask"` has no batch dimension,
unlike every other tensor in a network, so its first dimension is the
sequence length rather than the batch size.

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
-\> `PipeOpTorchTransformerEncoderLayer`

## Methods

### Public methods

- [`PipeOpTorchTransformerEncoderLayer$new()`](#method-PipeOpTorchTransformerEncoderLayer-initialize)

- [`PipeOpTorchTransformerEncoderLayer$clone()`](#method-PipeOpTorchTransformerEncoderLayer-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchTransformerEncoderLayer$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchTransformerEncoderLayer$new(
      id = "nn_transformer_encoder_layer",
      src_mask = FALSE,
      src_key_padding_mask = FALSE,
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `src_mask`:

  (`logical(1)`)  
  Whether the attention mask is provided as an additional input channel
  `"src_mask"`. This is a *construction* argument (and not a
  hyperparameter), because it determines the structure of the
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html). The
  default is `FALSE`, i.e. the layer attends over the full sequence. See
  section *Input and Output Channels* for more information.

- `src_key_padding_mask`:

  (`logical(1)`)  
  Whether the padding mask is provided as an additional input channel
  `"src_key_padding_mask"`. This is a *construction* argument (and not a
  hyperparameter), because it determines the structure of the
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html). The
  default is `FALSE`, i.e. no position is treated as padding. See
  section *Input and Output Channels* for more information.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchTransformerEncoderLayer$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchTransformerEncoderLayer$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = po("nn_transformer_encoder_layer", nhead = 4)
pipeop
#> 
#> ── PipeOp <nn_transformer_encoder_layer>: not trained ──────────────────────────
#> Values: nhead=4
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
#> <ParamSet(8)>
#>                 id    class lower upper nlevels        default  value
#>             <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:           nhead ParamInt     1   Inf     Inf <NoDefault[0]>      4
#> 2: dim_feedforward ParamInt     1   Inf     Inf           2048 [NULL]
#> 3:         dropout ParamDbl     0     1     Inf            0.1 [NULL]
#> 4:      activation ParamUty    NA    NA     Inf           relu [NULL]
#> 5:  layer_norm_eps ParamDbl     0   Inf     Inf          1e-05 [NULL]
#> 6:      norm_first ParamLgl    NA    NA       2          FALSE [NULL]
#> 7:            bias ParamLgl    NA    NA       2           TRUE [NULL]
#> 8:       is_causal ParamLgl    NA    NA       2          FALSE [NULL]
```
