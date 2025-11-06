# CLS Token for FT-Transformer

Concatenates a CLS token to the input as the last feature. The input
shape is expected to be `(batch, n_features, d_token)` and the output
shape is `(batch, n_features + 1, d_token)`.

This is used in the
[`LearnerTorchFTTransformer`](https://mlr3torch.mlr-org.com/reference/mlr_learners.ft_transformer.md).

## Usage

``` r
nn_ft_cls(d_token, initialization)
```

## Arguments

- d_token:

  (`integer(1)`)  
  The dimension of the embedding.

- initialization:

  (`character(1)`)  
  The initialization method for the embedding weights. Possible values
  are `"uniform"` and `"normal"`.

## References

Devlin, Jacob, Chang, Ming-Wei, Lee, Kenton, Toutanova, Kristina (2018).
“Bert: Pre-training of deep bidirectional transformers for language
understanding.” *arXiv preprint arXiv:1810.04805*.
