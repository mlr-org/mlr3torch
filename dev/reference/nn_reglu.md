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
#> -0.0000  0.3395  0.9543  0.6646 -0.0000
#> -0.0946 -0.0261 -0.0000  0.2921 -0.0000
#>  4.3130  0.0079  0.0000  0.0000 -0.0000
#> -1.0675  0.4318 -0.0000  0.2988 -0.0000
#> -0.0000 -0.0667  0.0000  1.7363  0.0000
#> -0.0000  0.0000 -0.0000  1.1446 -0.0663
#> -0.0000  1.2659  0.0000  0.9947 -0.0000
#>  0.0817  0.2484  0.0000 -0.0000  0.0000
#>  0.0055  0.0000 -1.2944  0.0000 -0.0000
#> -0.8235  3.0982 -1.0693  0.3252  0.0000
#> [ CPUFloatType{10,5} ]
```
