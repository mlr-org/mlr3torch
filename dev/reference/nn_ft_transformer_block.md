# Single Transformer Block for FT-Transformer

A transformer block consisting of a multi-head self-attention mechanism
followed by a feed-forward network.

This is used in
[`LearnerTorchFTTransformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md).

## Usage

``` r
nn_ft_transformer_block(
  d_token,
  attention_n_heads,
  attention_dropout,
  attention_initialization,
  ffn_d_hidden = NULL,
  ffn_d_hidden_multiplier = NULL,
  ffn_dropout,
  ffn_activation,
  residual_dropout,
  prenormalization,
  is_first_layer,
  attention_normalization,
  ffn_normalization,
  query_idx = NULL,
  attention_bias,
  ffn_bias_first,
  ffn_bias_second
)
```

## Arguments

- d_token:

  (`integer(1)`)  
  The dimension of the embedding.

- attention_n_heads:

  (`integer(1)`)  
  Number of attention heads.

- attention_dropout:

  (`numeric(1)`)  
  Dropout probability in the attention mechanism.

- attention_initialization:

  (`character(1)`)  
  Initialization of the query, key and value projections, either
  `"kaiming"` or `"xavier"`.

- ffn_d_hidden:

  (`integer(1)`)  
  Hidden dimension of the feed-forward network. Multiplied by 2 if using
  ReGLU or GeGLU activation.

- ffn_d_hidden_multiplier:

  (`numeric(1)`)  
  Alternative way to specify the hidden dimension of the feed-forward
  network as `d_token * d_hidden_multiplier`. Also multiplied by 2 if
  using RegLU or GeGLU activation.

- ffn_dropout:

  (`numeric(1)`)  
  Dropout probability in the feed-forward network.

- ffn_activation:

  (`nn_module`)  
  Activation function for the feed-forward network.

- residual_dropout:

  (`numeric(1)`)  
  Dropout probability for residual connections.

- prenormalization:

  (`logical(1)`)  
  Whether to apply normalization before attention and FFN (`TRUE`) or
  after (`FALSE`).

- is_first_layer:

  (`logical(1)`)  
  Whether this is the first layer in the transformer stack.

- attention_normalization:

  (`nn_module`)  
  Normalization module to use for attention.

- ffn_normalization:

  (`nn_module`)  
  Normalization module to use for the feed-forward network.

- query_idx:

  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)  
  Indices of the tensor to apply attention to. Should not be set
  manually. If NULL, then attention is applied to the entire tensor. In
  the last block in a stack of transformers, this is set to `-1` so that
  attention is applied only to the embedding of the CLS token.

- attention_bias:

  (`logical(1)`)  
  Whether attention has a bias.

- ffn_bias_first:

  (`logical(1)`)  
  Whether the first layer in the FFN has a bias.

- ffn_bias_second:

  (`logical(1)`)  
  Whether the second layer in the FFN has a bias.

## References

Devlin J, Chang M, Lee K, Toutanova K (2018). “Bert: Pre-training of
deep bidirectional transformers for language understanding.” *arXiv
preprint arXiv:1810.04805*. Gorishniy Y, Rubachev I, Khrulkov V, Babenko
A (2021). “Revisiting Deep Learning for Tabular Data.” *arXiv*,
**2106.11959**.
