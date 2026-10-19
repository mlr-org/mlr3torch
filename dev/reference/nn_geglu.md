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
#> -0.6416 -0.0547 -0.0822  0.2682  0.1295
#> -0.0420 -0.0481 -0.5612 -0.1204  0.1814
#>  0.0907 -1.3943  0.4909 -0.0890 -0.3728
#> -0.0130 -0.0035 -0.2438 -0.0891  0.0177
#> -0.0855 -0.0086  0.1588  0.0426  0.0112
#> -0.1359  0.7172 -0.4480  0.0067 -0.2025
#> -0.3552  0.0007 -0.1214 -0.0206 -0.0755
#> -0.3153 -0.1137  0.0091  0.1483 -0.0473
#> -1.8713 -0.0238  0.2033 -0.1066  1.1615
#> -0.0085 -0.1583  0.7378  0.1910  0.4090
#> [ CPUFloatType{10,5} ]
```
