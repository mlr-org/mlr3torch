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
#>  0.4500  0.0000  0.0987 -0.4315 -0.0000
#>  0.0012 -0.1616 -0.1417 -0.0000  0.0000
#>  0.0000 -0.0000 -0.0000  0.0000 -0.0752
#>  0.0000 -0.0000 -0.0000  0.0000  0.0000
#>  0.0000  0.0000  0.0000 -0.5223  0.0000
#> -0.8704 -0.7574  0.2717 -0.5101  0.0648
#> -0.0000  0.5129  0.0000  0.0000  0.0628
#> -0.0000  0.0000 -0.0000  0.0000  0.1255
#> -0.3347  0.8678  0.1680  0.3143  0.3141
#>  0.0000  0.6181 -0.0000  0.1252 -0.3556
#> [ CPUFloatType{10,5} ]
```
