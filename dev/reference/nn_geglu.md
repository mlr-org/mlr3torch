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
#>  0.5674  0.4685 -0.0742  0.0772  0.0093
#>  0.1418 -0.0859 -0.0568 -0.3289  0.0316
#>  0.0546  0.2050 -1.7980  0.0471 -0.2400
#> -0.0278  0.0936 -0.0185 -1.2179 -0.0016
#>  0.0881  0.4919  0.0134  0.0092 -0.6569
#> -0.0450  0.0254 -0.0418 -0.0664  0.0250
#> -0.0872 -0.2182 -0.1171 -1.7723 -0.1311
#> -0.7783  0.7483 -0.2110  0.1168  0.1742
#>  0.0514 -0.2825  0.0806  0.1254  0.6823
#>  0.3417  0.1094  0.0644 -0.0579 -0.7752
#> [ CPUFloatType{10,5} ]
```
