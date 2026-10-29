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
#>  0.0075 -0.0832 -1.8501  0.0947  0.0437
#> -0.0388  0.0259  0.0834  0.0065  0.0518
#> -0.0592 -0.0135  0.1566 -0.0161 -0.1022
#>  0.0163 -0.0071 -0.0160  0.0028 -0.8131
#>  0.0170 -0.1235  0.1173 -0.0537 -0.0972
#> -0.1960 -0.0904 -1.0001 -0.0012  0.0305
#>  0.0907 -0.0631 -0.0287 -0.0061  0.0214
#> -0.4626 -0.1928 -0.0009 -0.0510 -0.1330
#>  0.0744  0.0154  0.0746  1.1275  0.1283
#>  0.2408  0.0631  0.0940 -0.4314  0.2434
#> [ CPUFloatType{10,5} ]
```
