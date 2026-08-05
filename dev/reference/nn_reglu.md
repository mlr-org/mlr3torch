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
#> -0.0000 -1.2084  0.0000 -0.0000  0.0000
#>  0.0000  0.0000  0.0222 -0.0017 -0.0000
#> -0.0000 -1.6379 -1.6759  0.0000  0.2517
#>  0.5281 -0.0000 -0.0000 -0.0000 -0.1628
#>  0.0000  0.8238  0.0000  0.0000 -0.1034
#>  0.3866 -0.0000 -1.2741 -0.9522  0.5646
#>  2.3566  0.0000 -0.7421  0.1764 -0.0000
#> -0.1521  0.0000 -0.0000  0.6932 -0.5209
#> -0.0000  0.0000 -0.6965  0.4991 -0.0000
#>  0.7034 -0.1352  0.0000 -0.8243 -0.6738
#> [ CPUFloatType{10,5} ]
```
