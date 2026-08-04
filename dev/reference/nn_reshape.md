# Reshape

Reshape a tensor to the given shape.

## Usage

``` r
nn_reshape(shape)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `function()`)  
  The desired output shape, or a `function(shape)` that is called on the
  shape of the input tensor and returns it.
