# Numeric Tokenizer

Tokenizes numeric features into a dense embedding. For an input of shape
`(batch, n_features)` the output shape is
`(batch, n_features, d_token)`.

## Usage

``` r
nn_tokenizer_num(n_features, d_token, bias, initialization)
```

## Arguments

- n_features:

  (`integer(1)`)  
  The number of features.

- d_token:

  (`integer(1)`)  
  The dimension of the embedding.

- bias:

  (`logical(1)`)  
  Whether to use a bias.

- initialization:

  (`character(1)`)  
  The initialization method for the embedding weights. Possible values
  are `"uniform"` and `"normal"`.

## References

Gorishniy Y, Rubachev I, Khrulkov V, Babenko A (2021). “Revisiting Deep
Learning for Tabular Data.” *arXiv*, **2106.11959**.
