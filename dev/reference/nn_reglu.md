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
#>  0.4863 -0.4389  0.0000  0.0000 -0.1320
#>  0.0000  0.0000  0.1903 -0.3123 -0.0000
#> -0.0000  0.0000  1.0648 -0.0645 -0.6876
#>  0.0000  0.0000  0.1268  0.0000  0.0000
#> -0.0000 -0.0000  0.0000 -0.3513 -0.0000
#>  0.0000  0.1799  0.0000 -0.0000 -0.0000
#>  0.0000 -0.0000 -0.0000 -0.5300  0.0000
#>  0.0462  9.5871  0.0000  0.8253  0.0376
#> -0.0000 -0.1780  0.1768 -0.1232  0.5309
#>  0.0000 -0.9379 -0.7302  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
