# Assert a List of Shapes

Checks that `shapes` is a non-empty
[`list()`](https://rdrr.io/r/base/list.html) of valid shapes, see
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md).

## Usage

``` r
assert_shapes(
  shapes,
  coerce = TRUE,
  named = FALSE,
  null_ok = FALSE,
  unknown_batch = NULL
)
```

## Arguments

- shapes:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  A [`list()`](https://rdrr.io/r/base/list.html) of shapes.

- coerce:

  (`logical(1)`)  
  Whether to coerce the shapes to
  [`integer()`](https://rdrr.io/r/base/integer.html) if possible.

- named:

  (`logical(1)`)  
  Whether the shapes must be uniquely named.

- null_ok:

  (`logical(1)`)  
  Whether `NULL`, i.e. a wholly unknown shape, is valid.

- unknown_batch:

  (`logical(1)` \| `NULL`)  
  Whether the batch dimension **must** be unknown, i.e. `NA`. If left
  `NULL` (default), the first dimension can be `NA` or not.

## Value

The shapes, coerced to
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
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md)

## Examples

``` r
assert_shapes(list(c(NA, 3), c(NA, 5)))
#> [[1]]
#> [1] NA  3
#> 
#> [[2]]
#> [1] NA  5
#> 
```
