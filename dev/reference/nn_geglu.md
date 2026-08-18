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
#>  1.7003  0.0010 -0.0353  0.1999 -0.0188
#>  0.1410 -1.8156 -0.1149 -0.1286  0.2255
#>  0.0796  0.7994 -0.0197  0.0700  0.0446
#> -0.0486 -0.0817 -0.8270  0.0401 -0.1259
#> -0.2254  0.0281 -0.0509 -0.3848 -0.2454
#>  0.5879 -0.3328 -0.5079  2.1713 -0.2191
#> -0.0138 -0.6354 -0.0255  0.1607  0.0746
#>  0.1624 -0.0716  0.1157 -0.1706 -0.0200
#>  0.1714  0.0817 -0.1017 -0.0677 -0.1027
#>  0.0451 -0.4740  0.0809  0.2378  0.0586
#> [ CPUFloatType{10,5} ]
```
