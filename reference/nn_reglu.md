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
#> -0.2808 -0.0457 -0.0000  0.0000  0.0000
#>  0.1494 -0.0000 -0.4612 -0.0000 -0.0156
#>  0.0883  0.1742  0.0000 -1.7215  0.1368
#>  0.2248 -1.3621  0.2798  0.0000  0.0000
#>  0.2213 -0.0000 -0.5722 -0.3250 -0.0000
#> -0.6591  0.0000  0.0112  0.4518 -1.1921
#> -0.0000 -0.2938 -0.0000  1.8987 -0.0000
#> -0.3795  0.6037 -2.2842 -1.1327 -0.0000
#> -0.0749  0.0000  0.0000 -0.0000 -0.0540
#> -0.5467 -0.0000 -0.0000  0.0191 -0.0000
#> [ CPUFloatType{10,5} ]
```
