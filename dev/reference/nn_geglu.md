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
#>  0.0438 -0.0296 -1.6668 -0.0708  0.0024
#> -0.1015 -0.0323 -2.3313 -0.0546  0.1229
#>  0.0698  0.0005  0.0500  0.0214  0.3242
#> -1.2200 -0.3900 -0.0656  0.0107 -0.4685
#>  0.5169  0.0064 -0.0394  0.0278  0.3998
#> -0.4877 -0.0090  0.0702 -0.2249 -2.4330
#>  1.7999  0.0460  0.0447  0.3702 -0.0155
#>  0.1923 -0.1402 -0.7127 -0.0132  0.0366
#>  0.2399 -0.0967  0.7529  0.2487 -0.0819
#>  0.0463  0.0202  0.0627 -0.2040  0.0578
#> [ CPUFloatType{10,5} ]
```
