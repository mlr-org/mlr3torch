# Assert that a Dimension Exists

Rejects a `dim` parameter that does not address a dimension of `shape`.
Resolve negative indices with
[`resolve_dim()`](https://mlr3torch.mlr-org.com/dev/reference/shape_helpers.md)
first and pass both, so that the error message can report the value the
user actually specified.

## Usage

``` r
assert_dim_in_range(dim, true_dim, shape, id)
```

## Arguments

- dim:

  (`integer(1)`)  
  The dimension as the user specified it: it may count back from the
  last dimension. Only used for the error message.

- true_dim:

  (`integer(1)`)  
  The same dimension resolved to a positive index, see
  [`resolve_dim()`](https://mlr3torch.mlr-org.com/dev/reference/shape_helpers.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape that `true_dim` addresses.

- id:

  (`character(1)`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error message names.

## Value

The resolved dimension, invisibly.

## See also

Other Shape Assertions:
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
shape = c(NA, 3, 8, 8)
assert_dim_in_range(-1L, resolve_dim(-1L, shape), shape, id = "nn_squeeze")
try(assert_dim_in_range(-5L, resolve_dim(-5L, shape), shape, id = "nn_squeeze"))
#> Error : PipeOp 'nn_squeeze' cannot use 'dim' -5 for the input shape (NA,3,8,8), which has 4 dimension(s).
```
