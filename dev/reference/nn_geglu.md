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
#> -0.0055  0.0103  0.2593 -0.0451 -0.1633
#>  0.3764 -0.2232 -1.2089  0.0070 -0.2772
#>  0.4429 -0.0118 -0.9864  0.0488  0.0292
#> -1.2064  0.1168  1.0658 -0.0035 -0.0049
#> -0.0662 -0.2155  1.9810 -0.7349 -0.0160
#> -0.1756  0.5199 -0.3657  0.0634  0.2603
#> -0.1864 -0.6999 -0.0347 -0.2145  1.5218
#> -0.0265 -0.1250 -0.2422 -0.1192 -0.0043
#> -0.0054 -0.0371  1.8457  0.0931  0.8059
#> -0.1366  0.0852 -0.0530  0.0004  0.0495
#> [ CPUFloatType{10,5} ]
```
