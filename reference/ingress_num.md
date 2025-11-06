# Ingress Token for Numeric Features

Represents an entry point representing a tensor containing all numeric
([`integer()`](https://rdrr.io/r/base/integer.html) and
[`double()`](https://rdrr.io/r/base/double.html)) features of a task.

## Usage

``` r
ingress_num(shape = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)  
  Shape that `batchgetter` will produce. Batch-dimension should be
  included as `NA`.

## Value

[`TorchIngressToken`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md)
