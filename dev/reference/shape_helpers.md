# Helpers for Shape Inference

Helpers for writing the `private$.shapes_out()` method of a
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).
They all propagate unknown (`NA`) dimensions, see the "Shape Inference"
section of
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
for the shape conventions.

## Usage

``` r
shape_to_str(x)

broadcast_shapes(shapes, id)

resolve_dim(dim, shape, insert = FALSE)
```

## Arguments

- x:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \|
  [`list()`](https://rdrr.io/r/base/list.html) of
  [`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  The shape(s) to format. `NULL` stands for an unknown shape.

- shapes:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shapes, all with the same number of dimensions.

- id:

  (`character(1)`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error message names.

- dim:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The dimension(s) the operator addresses. Negative ones count back from
  the last dimension: `-1` is the last dimension, `-2` the one before
  it, and so on.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shape.

- insert:

  (`logical(1)`)  
  Whether a dimension is inserted rather than addressed, as by
  [`nn_unsqueeze()`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_unsqueeze.md):
  there is then one more position than the shape has dimensions, `-1`
  appending a new last one.

## Value

`broadcast_shapes()` returns an
[`integer()`](https://rdrr.io/r/base/integer.html) shape,
`resolve_dim()` an [`integer()`](https://rdrr.io/r/base/integer.html) of
the same length as `dim`, and `shape_to_str()` a `character(1)`.

## Details

- `broadcast_shapes()` applies the broadcasting rules of `torch`,
  generalized to shapes that may contain `NA`. Per dimension a known
  size that is not 1 wins; if all known sizes are 1 and some input is
  unknown, the result is unknown, because the unknown one may be greater
  than 1 and would then determine the size. The shapes must already have
  the same number of dimensions: shorter ones are not left-padded with
  1s, because the first dimension is the batch dimension.

- `resolve_dim()` resolves dimension indices that count from the end, as
  in `torch`, to positive ones. Indices that are out of range stay out
  of range, so that
  [`assert_dim_in_range()`](https://mlr3torch.mlr-org.com/dev/reference/assert_dim_in_range.md)
  reports them.

- `shape_to_str()` formats a shape, or a
  [`list()`](https://rdrr.io/r/base/list.html) of them, for an error
  message.

## See also

Other Shape Inference:
[`infer_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/infer_shapes.md),
[`reshape_output_shape()`](https://mlr3torch.mlr-org.com/dev/reference/reshape_output_shape.md)

## Examples

``` r
broadcast_shapes(list(c(NA, 1), c(NA, 6)), id = "nn_merge_sum")
#> [1] NA  6
resolve_dim(-1, c(NA, 3, 8))
#> [1] 3
shape_to_str(c(NA, 3, 8))
#> [1] "(NA,3,8)"
```
