# Assert that Shapes have the Same Number of Dimensions

Rejects inputs that do not all have the same number of dimensions, which
the operators that combine several inputs require: shorter shapes are
not left-padded with 1s, because the first dimension is the batch
dimension.

## Usage

``` r
assert_same_ndim(shapes, id)
```

## Arguments

- shapes:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shapes.

- id:

  (`character(1)`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error message names.

## Value

The shapes, invisibly.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
assert_same_ndim(list(c(NA, 3), c(NA, 5)), id = "nn_merge_cat")
try(assert_same_ndim(list(c(NA, 3), c(NA, 5, 2)), id = "nn_merge_cat"))
#> Error : PipeOp 'nn_merge_cat' requires all its inputs to have the same number of dimensions, but got the shapes [(NA,3);(NA,5,2)] (with 2, 3 dimensions).
```
