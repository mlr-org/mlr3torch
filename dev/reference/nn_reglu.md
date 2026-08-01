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
#>  0.0000 -0.0956 -0.0000  0.0000 -1.9168
#>  0.0000  0.0000 -0.0000 -0.0000  0.0000
#>  0.0000  0.7391 -0.0000  0.1524 -0.0000
#> -0.0000 -0.0000  0.0000 -0.0000  0.0000
#>  1.0675 -0.0884 -0.0000 -0.1823 -2.4762
#> -0.0000  0.0000 -1.0821 -1.1857  0.6124
#> -0.3153 -0.0000 -0.5766  0.1616  0.8453
#>  0.0043 -0.0000  0.0000  0.5419  0.4296
#> -0.3361 -0.0000  0.6416  0.0000 -0.0000
#> -0.0000  0.4388  0.6036  0.0529  0.0000
#> [ CPUFloatType{10,5} ]
```
