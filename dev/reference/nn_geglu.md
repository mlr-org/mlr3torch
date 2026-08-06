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
#> -0.5618 -0.0480  0.2823  0.2274 -0.1873
#> -0.2891 -0.2935 -0.1720  0.0448  0.2384
#> -0.0575 -0.5742 -0.0069 -0.0797 -0.0665
#> -0.0011  0.9105  0.0004  0.1134  1.0356
#> -0.0629 -0.2425  0.6383 -1.1657  0.6546
#>  0.1038  2.9167 -0.0765  0.0183 -0.1235
#> -0.6185  0.5158 -1.8845 -0.2207 -0.0547
#> -0.0315  0.3167 -1.8477  0.1523 -0.1935
#> -0.0215  0.0434 -0.9433  0.0475  0.2092
#>  0.1647  0.1286  1.1002 -0.0916  0.1834
#> [ CPUFloatType{10,5} ]
```
