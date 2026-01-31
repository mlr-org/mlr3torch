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
#>  1.6835  0.0000 -0.0000  0.1800  0.0000
#>  0.1097  0.0000  0.4043  0.0000  0.0000
#>  0.0000  0.0548 -0.0000  0.0000  0.0000
#> -0.0000 -0.0000  0.0000 -0.0000 -0.0000
#>  0.0000 -0.1519 -0.0003  0.1960  0.9549
#>  0.5942  0.0259  0.5867 -0.9073 -0.0000
#>  0.2793  0.0000  0.0232 -0.0000 -0.0940
#> -0.0000  0.0000 -0.0036  0.6939 -0.0000
#> -0.0000 -0.0050  0.0000  0.4301 -0.0095
#>  0.9530 -0.0000 -0.1258 -1.8613  0.1206
#> [ CPUFloatType{10,5} ]
```
