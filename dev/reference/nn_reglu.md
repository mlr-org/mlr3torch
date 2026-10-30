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
#>  0.0000 -0.4959 -0.1944  0.0000 -0.0000
#>  0.0000 -0.0253 -0.3440 -0.8128 -0.0000
#>  0.0000  0.0000 -1.4009  0.0000 -0.0000
#> -0.0517 -0.5303  0.0000 -0.5640  0.0000
#>  0.0040  0.3228  0.2747 -0.5517  0.0000
#> -0.3912 -0.0944 -0.0000 -0.8726 -0.0000
#>  0.0000 -0.2285  0.0000  0.0000  0.6851
#>  0.8268  1.7902  0.0327  0.0000 -0.4975
#> -0.0000  0.0000 -0.1671 -0.0000 -0.0000
#>  0.0000  0.0000 -0.1889 -0.0104 -0.0000
#> [ CPUFloatType{10,5} ]
```
