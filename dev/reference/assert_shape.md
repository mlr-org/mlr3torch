# Assert a Shape

Checks that `shape` is a valid shape, i.e. an
[`integer()`](https://rdrr.io/r/base/integer.html) with at least one
dimension, `NA` where a dimension is unknown. See the "Shape Inference"
section of
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
for the conventions.

## Usage

``` r
assert_shape(
  shape,
  null_ok = FALSE,
  coerce = TRUE,
  unknown_batch = NULL,
  len = NULL
)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  A shape, with `NA` for the dimensions whose size is unknown.

- null_ok:

  (`logical(1)`)  
  Whether `NULL`, i.e. a wholly unknown shape, is valid.

- coerce:

  (`logical(1)`)  
  Whether to coerce the input to an
  [`integer()`](https://rdrr.io/r/base/integer.html) if possible.

- unknown_batch:

  (`logical(1)` \| `NULL`)  
  Whether the batch dimension **must** be unknown, i.e. `NA`. If left
  `NULL` (default), the first dimension can be `NA` or not.

- len:

  (`integer(1)`)  
  The required number of dimensions.

## Value

The shape, coerced to an
[`integer()`](https://rdrr.io/r/base/integer.html) if `coerce` is
`TRUE`.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
assert_shape(c(NA, 3, 32, 32))
#> [1] NA  3 32 32
try(assert_shape("not a shape"))
#> Error : Invalid shape: must be an integer vector, but is character.
```
