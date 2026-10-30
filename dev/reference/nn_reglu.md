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
#> -0.0550 -0.0000  0.0883 -0.0000 -0.0000
#>  0.2395  0.0000 -0.0000 -0.0832  0.0000
#>  0.0271 -0.0000 -0.0000 -0.0000 -0.2451
#> -0.0000 -0.0000  0.6019  0.0000 -0.0000
#>  1.6408  1.0170 -0.0000 -0.0000 -0.8450
#> -0.1874 -0.2371 -0.1690 -0.0000  0.0000
#>  0.2734 -0.0000  0.0000  0.0000 -0.2891
#>  0.0017  0.0000 -0.0000  0.0000  1.2981
#>  0.0000  0.2728  0.5219 -0.1806  1.0086
#>  0.0000  0.0913  0.7508 -0.0000  0.0006
#> [ CPUFloatType{10,5} ]
```
