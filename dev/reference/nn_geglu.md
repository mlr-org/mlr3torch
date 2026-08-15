# GeGLU Module

This module implements the Gaussian Error Linear Unit Gated Linear Unit
(GeGLU) activation function. It computes \\\text{GeGLU}(x, g) = x \cdot
\text{GELU}(g)\\ where \\x\\ and \\g\\ are created by splitting the
input tensor in half along the last dimension.

## Usage

``` r
nn_geglu()
```

## References

Shazeer N (2020). “GLU Variants Improve Transformer.” 2002.05202,
<https://arxiv.org/abs/2002.05202>.

## Examples

``` r
x = torch::torch_randn(10, 10)
glu = nn_geglu()
glu(x)
#> torch_tensor
#>  0.0160  0.0859  1.0957 -0.4980  0.2456
#>  0.1417 -0.4060  0.1045 -0.0341  0.0264
#> -0.1851  0.0446  0.0623 -0.3602  0.4724
#>  0.1324  0.1101  1.1140  0.5564  1.1935
#> -0.0239  0.2628 -0.2316 -0.0146 -0.0511
#>  0.0168  0.2295 -0.3362  0.0491  0.0350
#> -0.0467  1.1569 -0.1067 -0.0852  0.7596
#> -0.1489  0.0387  0.0385 -0.0499  0.1069
#>  0.2430  0.0391  0.2354  0.1794  0.0168
#> -0.4723 -0.0245 -0.4706 -0.2808  0.0660
#> [ CPUFloatType{10,5} ]
```
