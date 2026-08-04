# Assert that a Dimension is not the Batch Dimension

Rejects an operation on the first dimension, which is the batch
dimension. An operator that changes it would silently change the number
of observations, which fails much later with a mismatch against the
target.

## Usage

``` r
assert_not_batch_dim(dim, shape, id)
```

## Arguments

- dim:

  (`integer(1)`)  
  The resolved dimension that the operator changes, see
  [`resolve_dim()`](https://mlr3torch.mlr-org.com/dev/reference/shape_helpers.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shape, used for the error message.

- id:

  (`character(1)`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error message names.

## Value

The dimension, invisibly.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
assert_not_batch_dim(2L, c(NA, 3, 8), id = "nn_squeeze")
try(assert_not_batch_dim(1L, c(NA, 3, 8), id = "nn_squeeze"))
#> Error : PipeOp 'nn_squeeze' would change dimension 1 of the input shape (NA,3,8), which is the batch dimension.
```
