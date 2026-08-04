# Assert that a Computed Output Size is Positive

Rejects a computed output size that no tensor can have, i.e. one that is
not a positive number. Operators such as convolutions and pooling
compute their spatial output sizes from `kernel_size`, `stride`,
`padding` and `dilation`, a combination of which can produce a size of
zero or less. Unknown (`NA`) sizes are accepted, since nothing can be
said about them.

## Usage

``` r
assert_positive_extent(extent, shape_in, id)
```

## Arguments

- extent:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  The sizes an operator computed for the dimensions it changes, e.g. the
  spatial dimensions of a convolution.

- shape_in:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shape, used for the error message.

- id:

  (`character(1)` \| `NULL`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error message names. `NULL` when the assertion is not made
  for a `PipeOp`.

## Value

The extent, invisibly.

## See also

Other Shape Assertions:
[`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md),
[`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md),
[`assert_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_ndim.md),
[`assert_not_batch_dim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_not_batch_dim.md),
[`assert_same_batch_size()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_batch_size.md),
[`assert_same_ndim()`](https://mlr3torch.mlr-org.com/dev/reference/assert_same_ndim.md),
[`assert_shape()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shape.md),
[`assert_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/assert_shapes.md)

## Examples

``` r
assert_positive_extent(c(30, NA), c(NA, 3, 32, 32), id = "nn_conv2d")
try(assert_positive_extent(c(0, 4), c(NA, 3, 32, 32), id = "nn_conv2d"))
#> Error : PipeOp 'nn_conv2d' cannot be applied to the input shape (NA,3,32,32): it would produce an output of size 0, 4, which no tensor can have. Check 'kernel_size', 'stride', 'padding' and 'dilation'.
```
