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
#>  0.0717  1.3870  0.0605  0.0066  0.1036
#>  0.0213 -0.1184  0.0214 -0.0877 -0.3012
#> -0.0393  0.0870  2.1058 -0.0196 -0.4575
#> -0.0242 -0.0598  0.8151 -0.0862 -0.2362
#> -0.1074 -0.0776  0.5130  0.2865 -0.1730
#> -0.1256 -0.0953 -1.7120 -0.0403 -0.0059
#> -0.0849  0.0274 -0.1092 -0.0249 -0.0030
#> -0.1700  0.2498 -0.0101  0.0358 -0.1614
#>  0.0742 -0.3554 -0.0103 -0.0431 -0.1758
#>  0.0232 -0.0593  0.0924 -0.0124 -0.0596
#> [ CPUFloatType{10,5} ]
```
