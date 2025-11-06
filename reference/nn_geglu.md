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
#> -0.0916  0.0575 -0.2261  0.1990  0.0387
#> -0.0302 -0.8046 -0.0642  0.0396 -0.2875
#> -0.0009 -1.8030 -0.0234  0.0294 -0.0066
#>  0.0075 -1.0522  0.0251  0.0014 -0.5944
#>  2.4198  0.0004  0.5740 -0.0562  2.4588
#> -0.1556 -0.8686  0.3393  0.2788  0.0618
#> -0.0941  0.1558  1.0937 -0.1316 -0.2059
#> -2.5652 -0.3479 -0.0057 -2.3589 -0.2496
#>  0.0249 -0.1046  0.1820 -0.1638  0.8881
#> -0.9601 -0.1820  0.6071 -0.0050 -0.0385
#> [ CPUFloatType{10,5} ]
```
