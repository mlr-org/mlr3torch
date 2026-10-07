# GeGLU Module

This module implements the Gaussian Error Linear Unit Gated Linear Unit
(GeGLU) activation function. It computes \\\text{GeGLU}(x, g) = x \cdot
\text{GELU}(g)\\ where \\x\\ and \\g\\ are created by splitting the
input tensor in half along the last dimension.

## Usage

``` r
nn_geglu()
```

## References

Shazeer N (2020). “GLU Variants Improve Transformer.” 2002.05202,
<https://arxiv.org/abs/2002.05202>.

## Examples

``` r
x = torch::torch_randn(10, 10)
glu = nn_geglu()
glu(x)
#> torch_tensor
#> -0.0115  0.0456 -0.0158 -0.0750 -0.1757
#>  1.7041 -0.0067 -0.1841 -0.0226 -0.2467
#>  0.0660 -1.0448  0.0042  0.7534  0.0060
#>  0.0141  0.0135 -0.4711  0.0122  0.2629
#>  0.6062 -1.2779 -0.2681 -0.0239 -0.2480
#>  0.0555  0.6232  0.5361  0.0273 -0.2139
#> -0.2475  0.0530  0.0198  0.0095  0.0876
#> -0.0179  0.0392  0.2380  0.1026  4.0513
#>  0.0044 -0.0483  0.0630 -0.2394  1.8226
#>  0.0431  0.0570 -0.1546 -0.1562  0.2333
#> [ CPUFloatType{10,5} ]
```
