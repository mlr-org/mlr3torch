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
#> -1.5162 -0.8909 -0.1834  0.0209  0.0548
#> -1.2575 -0.0789 -0.1747 -0.0152 -0.5659
#> -0.0712  0.6054  0.0269 -0.1307 -0.6146
#>  0.0305  0.0678 -0.0370  0.0197  0.0861
#>  0.0546  0.0017 -0.0032  0.0064  0.1034
#> -0.3283 -1.3140 -0.3709 -0.0793 -0.1798
#>  0.3858  0.0966 -1.7099 -1.1025 -2.6420
#> -0.0754  0.1021  0.0390 -0.0083  0.0308
#> -0.0298 -0.0339  0.0944  0.2624 -0.1063
#>  0.0906  0.5731 -0.2594 -0.1727  0.0138
#> [ CPUFloatType{10,5} ]
```
