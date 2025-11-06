# Shape of Lazy Tensor

Shape of a lazy tensor. Might be `NULL` if the shapes is not known or
varying between rows. Batch dimension is always `NA`.

## Usage

``` r
lazy_shape(x)
```

## Arguments

- x:

  ([`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md))  
  Lazy tensor.

## Value

([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)

## Examples

``` r
lt = as_lazy_tensor(1:10)
lazy_shape(lt)
#> [1] NA  1
lt = as_lazy_tensor(matrix(1:10, nrow = 2))
lazy_shape(lt)
#> [1] NA  5
```
