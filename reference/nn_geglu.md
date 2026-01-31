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
#> -1.9972e-01  9.4353e-02  9.1621e-03 -1.0869e-01 -6.4350e-02
#> -2.1930e-01 -2.1756e-03  8.4930e-02  3.4412e-01  2.9060e+00
#> -2.1002e-03 -5.7275e-01 -1.0464e-01 -4.5219e-02 -4.5341e-01
#>  2.0871e+00  8.3240e-01  3.3760e-01 -8.4155e-02  1.3400e+00
#>  2.8810e-02  8.4222e-03 -4.5405e-01  1.2029e+00 -9.7085e-06
#> -4.7302e-02 -1.0333e-01  1.2813e+00  2.8165e-02  1.3820e-01
#> -3.8109e-02  2.8558e-02  2.6088e-01 -5.9199e-01  1.4955e-01
#> -2.6725e-01  7.7697e-03 -1.4376e-02 -4.1632e-02  7.2377e-03
#> -1.0878e-01 -1.6593e-02 -1.8377e-01 -8.4827e-03 -7.3444e-01
#>  1.4560e-01  2.6953e-03  3.1963e-01 -4.1641e-02  8.5725e-02
#> [ CPUFloatType{10,5} ]
```
