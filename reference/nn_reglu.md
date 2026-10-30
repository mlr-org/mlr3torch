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
#> -0.0479 -1.7395 -0.0450 -0.0708 -0.0000
#> -0.2064 -0.5957  0.0000 -0.2919 -0.8651
#> -0.0455  0.3419  0.2988 -0.7240 -1.0787
#>  0.0000 -0.4891 -0.0000 -0.0000 -0.0000
#>  0.0000 -0.3980  0.0000 -0.0000 -0.0181
#>  0.0000 -0.4482 -0.1623 -0.0000 -0.0108
#> -0.0000 -0.0000 -0.0000 -0.0147  0.0000
#> -0.4978 -0.0000 -0.0000 -0.8918 -0.0000
#> -0.0000 -0.0000 -0.2104  0.0000 -0.2950
#>  0.2257 -0.0465 -0.0000 -0.0000 -0.0394
#> [ CPUFloatType{10,5} ]
```
