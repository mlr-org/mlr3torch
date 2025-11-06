# Compare lazy tensors

Compares lazy tensors using their indices and the data descriptor's
hash. This means that if two
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)s:

- are equal: they will mateterialize to the same tensors.

- are unequal: they might materialize to the same tensors.

## Usage

``` r
# S3 method for class 'lazy_tensor'
x == y
```

## Arguments

- x, y:

  ([`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md))  
  Values to compare.
