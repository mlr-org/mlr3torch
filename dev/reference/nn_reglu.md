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
#>  0.0000 -0.9617 -0.4168  0.0000 -0.0000
#> -0.0000  1.8041  0.7191 -1.0841  0.0000
#> -2.1416  0.4791  0.0000 -0.3419  0.1490
#>  0.0000 -0.0000 -0.0000  0.0379 -0.0000
#>  0.4429  0.0000  0.0000 -0.0000  0.0000
#>  0.7088  0.0024 -0.0000  3.3407 -0.1185
#>  0.0000  1.9878 -0.0559 -0.9331  0.4927
#> -0.0000 -0.0000 -0.3850  2.3700 -0.0000
#>  0.6937  0.0020  0.0000 -0.0000  2.4001
#>  0.1307  0.0000 -0.0000 -0.0000 -0.6265
#> [ CPUFloatType{10,5} ]
```
