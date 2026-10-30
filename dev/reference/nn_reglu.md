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
#> -0.2712  0.0868 -2.6695  0.0574 -0.1464
#>  0.0757 -3.5777 -0.0000 -0.0000 -0.0000
#>  0.6968  0.4147  0.2723 -0.0000 -0.0000
#>  0.0000 -1.2054 -0.0000 -0.0000  0.0000
#>  1.8876 -0.0000  0.0000  0.6246  0.0000
#> -0.0000 -0.6395 -0.0887  0.0000  0.0000
#> -0.0000 -0.0000  0.0000  0.0000  0.0000
#> -0.0000  1.1706 -0.2722  1.0529  0.0000
#>  0.9294  0.0000 -0.0000 -0.0000 -0.0000
#>  0.0000  0.0000 -0.0649  0.0000  0.4292
#> [ CPUFloatType{10,5} ]
```
