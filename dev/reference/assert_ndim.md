# Assert the Number of Dimensions of a Shape

Rejects a shape with the wrong number of dimensions. Give either `ndim`,
the number(s) of dimensions the operator accepts, or the bounds `min`
and `max` (either on its own), which is what an operator that accepts a
range of them uses, as batch normalization does.

## Usage

``` r
assert_ndim(shape, ndim = NULL, id, min = NULL, max = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shape, with `NA` for the dimensions whose size is unknown.

- ndim:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The number(s) of dimensions the operator accepts, the batch dimension
  included. Alternatively give `min` and/or `max`.

- id:

  (`character(1)`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error message names.

- min:

  (`integer(1)`)  
  The smallest number of dimensions the operator accepts, `NULL` for no
  lower bound.

- max:

  (`integer(1)`)  
  The largest number of dimensions the operator accepts, `NULL` for no
  upper bound.

## Value

The shape, invisibly.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
assert_ndim(c(NA, 3, 32, 32), 4L, id = "nn_conv2d")
try(assert_ndim(c(NA, 3), 4L, id = "nn_conv2d"))
#> Error : PipeOp 'nn_conv2d' requires an input with 4 dimensions (the first one being the batch dimension), but got the shape (NA,3), which has 2.
# batch normalization accepts a range
try(assert_ndim(c(NA, 3, 4, 5), id = "nn_batch_norm1d", min = 2L, max = 3L))
#> Error : PipeOp 'nn_batch_norm1d' requires an input with 2 or 3 dimensions (the first one being the batch dimension), but got the shape (NA,3,4,5), which has 4.
```
