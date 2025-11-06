# Create a lazy tensor

Create a lazy tensor.

## Usage

``` r
lazy_tensor(data_descriptor = NULL, ids = NULL)
```

## Arguments

- data_descriptor:

  ([`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)
  or `NULL`)  
  The data descriptor or `NULL` for a lazy tensor of length 0.

- ids:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The elements of the `data_descriptor` to be included in the lazy
  tensor.

## Examples

``` r
ds = dataset("example",
  initialize = function() self$iris = iris[, -5],
  .getitem = function(i) list(x = torch_tensor(as.numeric(self$iris[i, ]))),
  .length = function() nrow(self$iris)
)()
dd = as_data_descriptor(ds, list(x = c(NA, 4L)))
lt = as_lazy_tensor(dd)
```
