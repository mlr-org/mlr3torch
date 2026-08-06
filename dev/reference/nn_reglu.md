# ReGLU Module

Rectified Gated Linear Unit (ReGLU) module. Computes the output as
\\\text{ReGLU}(x, g) = x \cdot \text{ReLU}(g)\\ where \\x\\ and \\g\\
are created by splitting the input tensor in half along the last
dimension.

## Usage

``` r
nn_reglu()
```

## References

Shazeer N (2020). “GLU Variants Improve Transformer.” 2002.05202,
<https://arxiv.org/abs/2002.05202>.

## Examples

``` r
x = torch::torch_randn(10, 10)
reglu = nn_reglu()
reglu(x)
#> torch_tensor
#> -0.0097  0.0000 -0.7260 -1.1384 -0.0000
#> -0.0000  0.0000 -0.0000  0.7515 -0.5065
#>  1.6251 -0.0000 -0.4315 -0.2775  0.0000
#>  0.0000  0.0248  0.0000 -0.0000  0.1607
#>  0.0000 -0.0000 -0.3766  0.3544  0.0000
#>  0.6108  0.0109 -0.0046 -0.6856 -0.0000
#>  0.0000  0.0954  0.0000 -0.0000 -0.0000
#>  0.0000 -0.3448 -0.0000  0.0000  0.0000
#>  0.1429  0.0724 -2.2538  0.0332 -0.0000
#> -0.0000 -2.2954 -0.1636  0.0918  0.1326
#> [ CPUFloatType{10,5} ]
```
