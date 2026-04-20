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
#> -0.0598 -0.0000  0.4781 -0.0958 -0.4001
#> -0.0000  1.0530 -0.0694  0.3661 -0.5631
#>  0.0000 -0.0000  0.5293 -0.0000  1.5019
#> -0.0000 -0.0000 -0.0000 -0.0000  0.2991
#>  0.0000  0.3512 -0.1334  0.0738 -0.0000
#>  0.0000  1.2444 -2.1510 -0.0000 -0.0000
#>  0.0000 -0.0000 -0.0000 -0.2377  2.2618
#> -0.0000  0.0000 -0.0000 -0.0000  1.2291
#> -0.0000 -0.2526 -0.7380 -0.0000 -0.0000
#> -0.0000  0.0000 -0.0511 -0.5425 -0.0000
#> [ CPUFloatType{10,5} ]
```
