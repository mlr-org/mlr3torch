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
#>  0.0000 -0.0000 -0.2267  0.6484 -0.0000
#> -0.2264 -0.0000  0.7444  0.0013 -0.0000
#>  0.0000 -0.0000  0.0000  0.0000  0.0000
#>  0.7168  0.4179  0.1762  2.1710 -0.7265
#> -0.6733  0.0000  0.0000  0.0000 -0.0000
#> -0.0000 -0.0000 -0.0404 -0.4064  0.0000
#> -3.3598 -0.0000 -0.0000 -0.3268  1.0215
#> -0.0000 -0.0000 -0.0000  0.0000 -0.0000
#>  0.0000 -1.0205  1.2490  0.0000 -0.4931
#> -0.0000 -0.0000 -0.0000 -0.3555  1.1504
#> [ CPUFloatType{10,5} ]
```
