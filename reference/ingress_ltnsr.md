# Ingress Token for Lazy Tensor Feature

Represents an entry point representing a tensor containing a single lazy
tensor feature.

## Usage

``` r
ingress_ltnsr(feature_name = NULL, shape = NULL)
```

## Arguments

- feature_name:

  (`character(1)`)  
  Which lazy tensor feature to select if there is more than one.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)  
  Shape that `batchgetter` will produce. Batch-dimension should be
  included as `NA`.

## Value

[`TorchIngressToken`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md)
