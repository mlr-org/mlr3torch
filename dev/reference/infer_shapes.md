# Infer Shapes

Infer the shapes of the output of a function based on the shapes of the
input. This works by running the function on the input and observing the
results. For fully known input shapes this is always correct. For
partially unknown shapes, the `NA`s are replaced with various concrete
values and the output shape is computed from them. Note that this is a
heuristic that might fail, so usually one wants to provide the shape
(inference) explicitly.

## Usage

``` r
infer_shapes(shapes_in, param_vals, output_names, fn, rowwise, id)
```

## Arguments

- shapes_in:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  A list of shapes of the input tensors.

- param_vals:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  A list of named parameters for the function.

- output_names:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The names of the output tensors.

- fn:

  (`function()`)  
  The function to infer the shapes for.

- rowwise:

  (`logical(1)`)  
  Whether the function is rowwise.

- id:

  (`character(1)`)  
  The id of the PipeOp (for error messages).

## Value

([`list()`](https://rdrr.io/r/base/list.html))  
A list of shapes of the output tensors.

## Details

The inference is done as follows:

1.  All `NA`s are replaced with three different values, which span a
    wide range: none of them is `1` (which broadcasts and is squeezed
    away), one of them is small (to detect operators that clamp to the
    input size, such as slicing or cropping) and the others are large
    (because operators such as a convolution with a large kernel need a
    minimum extent).

2.  Three tensors are generated for the three shapes of step 1.

3.  The function is called on these three tensors and the shapes are
    calculated. A call that fails is dropped, so that an operator is not
    rejected because of the smallest value; at least two of the three
    calls must succeed.

4.  If:

    - the number of dimensions varies, an error is thrown.

    - the number of dimensions is the same, values are set to `NA` if
      the dimension is varying between the tensors and otherwise set to
      the unique value.

## See also

Other Shape Inference:
[`reshape_output_shape()`](https://mlr3torch.mlr-org.com/dev/reference/reshape_output_shape.md),
[`shape_to_str()`](https://mlr3torch.mlr-org.com/dev/reference/shape_helpers.md)
