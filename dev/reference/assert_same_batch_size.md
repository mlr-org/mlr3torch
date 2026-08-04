# Assert that Shapes have the Same Batch Size

Rejects inputs whose batch sizes, i.e. first dimensions, disagree. An
unknown (`NA`) batch size is compatible with any other, so only the
known ones have to agree.

Unlike the other assertions this returns the common batch size rather
than its input, because that is what the caller needs next: an operator
that drops the batch dimension while it works has to put it back
afterwards.

## Usage

``` r
assert_same_batch_size(shapes, id)
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

(`integer(1)`) The common batch size, invisibly, or `NA_integer_` if no
input has a known one.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_positive_extent()`](https://mlr3torch.mlr-org.com/dev/reference/assert_positive_extent.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
assert_same_batch_size(list(c(8, 3), c(8, 5)), id = "nn_block")
# an unknown batch size is compatible with a known one, which is the one that is returned
assert_same_batch_size(list(c(NA, 3), c(8, 5)), id = "nn_block")
try(assert_same_batch_size(list(c(8, 3), c(4, 5)), id = "nn_block"))
#> Error : PipeOp 'nn_block' requires all its inputs to have the same batch size, but got the shapes [(8,3);(4,5)] (with the batch sizes 8 and 4).
```
