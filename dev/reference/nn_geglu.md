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
#> -0.9860 -0.1299  0.0027  0.8386  0.0645
#> -0.0003  0.0062  0.0262  1.9573  0.2794
#>  0.0253 -0.0795 -0.2882  0.0139  1.0735
#>  0.0342  0.0909  0.0328 -0.2708  0.0870
#>  0.0896 -1.6359  0.1308  0.2124  0.0517
#> -0.0084  0.5372 -1.9560 -0.0741  0.0206
#>  0.0868 -0.0067  0.0015  0.1312  0.0057
#>  0.9041  0.1626  0.8268  0.0431 -1.8653
#> -0.0449 -0.0935  0.2053 -0.0315  0.0474
#>  0.0704 -0.2445  0.0363 -0.0117  0.1470
#> [ CPUFloatType{10,5} ]
```
