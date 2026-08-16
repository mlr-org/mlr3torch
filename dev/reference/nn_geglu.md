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
#> -0.2373 -0.0322 -0.3867  0.2286 -0.0327
#> -0.0097 -0.5731 -0.0979  0.1003  0.0025
#>  0.0143 -0.7432 -0.1558  0.0572  0.0863
#>  0.0669 -3.9893 -0.0303  0.0550 -0.2494
#>  1.5545  0.0742  0.0527 -0.0421  0.0836
#> -0.9761  0.0957 -1.0651  1.3075 -0.7057
#> -0.0971 -0.0248  0.0608 -0.0378  0.0603
#>  0.0100 -0.4280  1.3560  0.5751  0.8316
#>  0.4063  0.6135  2.1378  0.9874 -0.7153
#> -0.0314 -0.0076  0.0209  0.8973  0.8809
#> [ CPUFloatType{10,5} ]
```
