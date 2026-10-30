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
#>  2.6432e-01 -6.5779e-02  9.6438e-05  2.1870e-02 -3.0953e-01
#> -1.9167e-01  2.4336e-02  4.4559e-01  8.1759e-02 -3.8840e-01
#>  6.8213e-02 -4.8431e-02 -2.8230e-01  3.1460e-02  1.0474e-01
#> -1.3191e+00 -1.6823e-01 -2.4453e-01  5.3803e-02  1.3283e-01
#> -7.6123e-01 -1.0420e+00  5.7896e-02  3.4428e-02 -3.7097e-02
#>  9.3492e-04 -7.0098e-01  8.7623e-03 -7.6976e-01 -2.3077e-01
#>  5.5934e-02 -1.3855e+00 -2.9775e-02  1.0839e-01  3.3119e-01
#> -1.8112e-01 -2.6469e-01 -1.2652e-01  8.3165e-02 -1.6830e-01
#>  5.5245e-01 -1.4820e-01 -1.0767e+00 -2.3434e-02  7.3383e-04
#> -1.7151e-01 -6.1896e-02  7.1306e-01  2.9800e-02 -1.2765e-01
#> [ CPUFloatType{10,5} ]
```
