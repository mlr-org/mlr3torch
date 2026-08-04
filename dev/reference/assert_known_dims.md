# Assert that Dimensions are Known

Ensure that a specific dimension is known, i.e. not `NA`.

## Usage

``` r
assert_known_dims(shape, dims, what, id = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shape.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Indices of the dimensions that must be known.

- what:

  (`character(1)`)  
  Describes those dimensions in the error message, e.g.
  `"the channel dimension (dimension 2)"`.

- id:

  (`character(1)` \| `NULL`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
  the assertion is made for, which the error message names. `NULL` when
  the assertion is not made for a `PipeOp`.

## Value

The shape, invisibly.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
# a convolution needs the number of input channels, but not the spatial extent
assert_known_dims(c(NA, 3, NA, NA), 2, "the number of channels", id = "nn_conv2d")
try(assert_known_dims(c(NA, NA, 10, 10), 2, "the number of channels", id = "nn_conv2d"))
#> Error : PipeOp 'nn_conv2d' requires the number of channels of the input shape to be known, but got shape (NA,NA,10,10).
```
