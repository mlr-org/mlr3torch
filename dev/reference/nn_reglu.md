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
#>  0.0000  0.0288 -0.0000 -0.6700 -0.0000
#> -0.2931  1.1636 -0.3938 -0.0000  0.1664
#>  0.2294 -0.0000 -0.0000 -0.0000 -0.0000
#> -0.0000  0.6701  0.0143  0.1294  0.6794
#>  0.0000  0.3997  0.0000 -0.3547  0.0000
#> -0.0000  0.0000 -0.0000  1.2896 -0.0000
#> -0.1710 -0.0000 -2.2725 -0.1124 -0.0000
#>  0.0000  0.9135  1.4874 -0.0000  0.0000
#> -0.0389  0.3907 -1.4279  0.6440 -0.0008
#>  0.0000  0.6778 -0.1971  0.3391  0.3992
#> [ CPUFloatType{10,5} ]
```
