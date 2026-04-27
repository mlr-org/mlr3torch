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
#>  0.6495 -0.1737 -0.3298 -0.0956  3.9962
#> -0.3033 -0.0828  0.0255 -0.2110  0.2820
#>  0.1101 -0.0345  0.0906  0.0237  0.6610
#>  0.2594 -0.0317 -0.0144  0.5091  0.0493
#>  0.2920  0.1341 -0.1746 -0.0837  0.1303
#> -0.1125  0.1996 -0.1414 -0.1955  0.4050
#>  0.1976  0.0156 -1.2533  0.0505  0.1869
#>  0.0814 -0.2641  0.2353 -0.0133 -0.2164
#> -0.0022 -0.1550  0.0560  0.0026 -0.0024
#>  0.1940 -0.0388  0.0698  0.0216 -2.2547
#> [ CPUFloatType{10,5} ]
```
