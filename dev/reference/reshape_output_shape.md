# Output Shape of a Reshape

The shape of
[`torch_reshape(x, shape)`](https://torch.mlverse.org/docs/reference/torch_reshape.html),
which is what
[`nn_reshape()`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_reshape.md)
infers. The dimension that `torch` infers from the number of elements is
resolved here whenever that number is known, and stays unknown
otherwise.

## Usage

``` r
reshape_output_shape(shape_in, shape, id)
```

## Arguments

- shape_in:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shape, with `NA` for the dimensions whose size is unknown.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `function()`)  
  The target shape, where `-1` marks the dimension that `torch` infers
  from the number of elements. A `function(shape)` of the input shape is
  called on it and must return such a vector, which lets a reshape be
  expressed for inputs whose sizes are not known in advance, e.g.
  `\(shape) c(shape[1:2], 10)`.

- id:

  (`character(1)`)  
  The id of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html),
  which the error messages name.

## Value

([`integer()`](https://rdrr.io/r/base/integer.html)) The output shape.

## See also

Other Shape Inference:
[`infer_shapes()`](https://mlr3torch.mlr-org.com/dev/reference/infer_shapes.md),
[`shape_to_str()`](https://mlr3torch.mlr-org.com/dev/reference/shape_helpers.md)

## Examples

``` r
reshape_output_shape(c(NA, 3, 4), c(-1, 12), id = "nn_reshape")
#> [1] NA 12
reshape_output_shape(c(NA, 3, 4), \(shape) c(shape[1], 12), id = "nn_reshape")
#> [1] NA 12
```
