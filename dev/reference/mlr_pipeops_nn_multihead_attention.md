# Multi-Head Attention

Multi-head attention as described in *Attention Is All You Need*.

This is a thin wrapper around
[`torch::nn_multihead_attention()`](https://torch.mlverse.org/docs/reference/nn_multihead_attention.html)
that makes it usable as a building block of a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) of
tensor operations, where both self-attention and cross-attention can be
expressed, see section *Input and Output Channels*.

## nn_module

Calls
[`torch::nn_multihead_attention()`](https://torch.mlverse.org/docs/reference/nn_multihead_attention.html)
when trained, where the parameters `embed_dim`, `kdim` and `vdim` are
inferred as the last dimension of the query, key and value tensors
respectively.

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

- `batch_first` :: `logical(1)`  
  Whether the input and output tensors are provided as
  `(batch, sequence, feature)` (`TRUE`) or as
  `(sequence, batch, feature)` (`FALSE`). Default is `FALSE`, as in
  `torch`.

- `avg_weights` :: `logical(1)`  
  Whether the returned attention weights are averaged over the attention
  heads. Default is `TRUE`. Only has an effect when the construction
  argument `need_weights` is `TRUE`.

Note that `embed_dim`, `kdim` and `vdim` are *not* parameters, as they
are inferred from the shapes of the input tensors.

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

Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones,
Llion, Gomez, N A, Kaiser, Łukasz, Polosukhin, Illia (2017). “Attention
is all you need.” *Advances in neural information processing systems*,
**30**.

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3torch::PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchMultiheadAttention`

## Methods

### Public methods

- [`PipeOpTorchMultiheadAttention$new()`](#method-PipeOpTorchMultiheadAttention-new)

- [`PipeOpTorchMultiheadAttention$clone()`](#method-PipeOpTorchMultiheadAttention-clone)

------------------------------------------------------------------------

### Method `new()`

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

### Method `clone()`

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
#> <ParamSet(7)>
#>               id    class lower upper nlevels        default  value
#>           <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:     num_heads ParamInt     1   Inf     Inf <NoDefault[0]>      4
#> 2:       dropout ParamDbl     0     1     Inf              0 [NULL]
#> 3:          bias ParamLgl    NA    NA       2           TRUE [NULL]
#> 4:   add_bias_kv ParamLgl    NA    NA       2          FALSE [NULL]
#> 5: add_zero_attn ParamLgl    NA    NA       2          FALSE [NULL]
#> 6:   batch_first ParamLgl    NA    NA       2          FALSE [NULL]
#> 7:   avg_weights ParamLgl    NA    NA       2           TRUE [NULL]
```
