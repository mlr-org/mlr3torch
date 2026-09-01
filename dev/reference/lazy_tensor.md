# Create a lazy tensor

Creates a `lazy_tensor` vector. Because it is a vector, it can be stored
in a `data.table`, which gives mlr3torch the ability to use arbitrary
tensors in its task. It is 'lazy', because the tensors are not stored
in-memory, but only loaded when calling
[`materialize()`](https://mlr3torch.mlr-org.com/dev/reference/materialize.md).
The vector itself only describes *how* to load the data. It is also
possible to preprocess lazy_tensors, e.g. via `po("augment_<key>")`, and
`po("trafo_<key>")`.

## Usage

``` r
lazy_tensor(data_descriptor = NULL, ids = NULL)
```

## Arguments

- data_descriptor:

  ([`DataDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/DataDescriptor.md)
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
