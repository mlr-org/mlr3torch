# Ingress Token for Categorical Features

Represents an entry point representing a tensor containing all
categorical ([`factor()`](https://rdrr.io/r/base/factor.html),
[`ordered()`](https://rdrr.io/r/base/factor.html),
[`logical()`](https://rdrr.io/r/base/logical.html)) features of a task.

## Usage

``` r
ingress_categ(shape = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)  
  Shape that `batchgetter` will produce. Batch-dimension should be
  included as `NA`.

## Value

[`TorchIngressToken`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md)
