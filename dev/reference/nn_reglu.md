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
#>  0.1164  1.4649 -0.1190 -0.0000 -0.0000
#> -0.4703  0.4840  0.8642 -0.0000  3.0353
#> -0.1010  0.0000 -0.0352  0.9725  0.0000
#> -0.0000  0.2325 -0.8662  0.0000  0.0000
#> -0.1436  0.0000 -0.0000 -0.1497  0.2237
#>  0.0000  0.0000  0.0944  0.8830  0.2178
#> -0.2605 -0.2749 -0.0000  0.5373  0.0000
#> -0.0000 -0.0000 -0.0000  0.2927 -0.0000
#> -0.0000  0.5110  0.0000 -0.0000  0.3729
#>  0.0000 -0.0000  1.1452  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
