# Infer Shapes

Infer the shapes of the output of a function based on the shapes of the
input. This is done as follows:

1.  All `NA`s are replaced with values `1`, `2`, `3`.

2.  Three tensors are generated for the three shapes of step 1.

3.  The function is called on these three tensors and the shapes are
    calculated.

4.  If:

    - the number of dimensions varies, an error is thrown.

    - the number of dimensions is the same, values are set to `NA` if
      the dimension is varying between the three tensors and otherwise
      set to the unique value.

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
