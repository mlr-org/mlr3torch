# Categorical Tokenizer

Tokenizes categorical features into a dense embedding. For an input of
shape `(batch, n_features)` the output shape is
`(batch, n_features, d_token)`.

## Usage

``` r
nn_tokenizer_categ(cardinalities, d_token, bias, initialization)
```

## Arguments

- cardinalities:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The number of categories for each feature.

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
