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
#>  7.1862e-02 -3.9657e-01 -5.3551e-02 -4.6575e-05  3.2440e-02
#>  6.5794e-01  4.0010e-01 -1.8819e-01  1.5603e-01 -4.4718e-01
#>  8.9538e-02 -4.3167e-02  1.0288e-01  1.3465e-01  7.4283e-02
#> -7.7661e-02  3.5814e-01 -2.9318e-02  8.5769e-02  1.0095e+00
#> -1.9177e+00  1.3196e+00 -7.9653e-03  7.0801e-02 -1.3474e-01
#>  9.9804e-02  1.7299e-01  1.1078e+00  8.0375e-02 -5.3425e-03
#> -1.3614e-01  3.2974e-03 -4.1148e-02  9.2145e-02  2.4369e-01
#>  1.1555e+00 -3.3282e-03  1.6350e-01 -1.0322e-02  5.0363e-02
#> -2.0393e+00  1.2934e+00 -2.4387e-01  6.8370e-01 -5.7927e-02
#> -1.0682e+00 -6.7164e-02  5.3258e-02 -1.2137e-01 -2.0619e-01
#> [ CPUFloatType{10,5} ]
```
