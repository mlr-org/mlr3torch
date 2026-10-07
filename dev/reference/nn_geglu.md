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
#>  0.0013 -0.5657  0.1632 -0.2792 -0.0061
#> -0.7954 -0.6304 -0.4211  0.0092  0.1667
#> -0.0335 -1.2665 -0.0674 -0.0235 -0.0774
#>  0.0241 -0.7878  0.1386 -0.2197  1.0184
#>  0.0930 -0.0709 -0.1225  0.0517  0.0930
#>  0.0734  0.0767  0.0299  0.0046 -0.0478
#>  0.3762 -0.0037 -0.0154 -0.0098 -0.0795
#> -0.2840  0.0239 -0.0633  0.0438  2.1801
#>  0.0833  0.6614  0.0112 -0.0081 -0.4254
#> -0.1326 -0.3485  0.1121  0.0186  0.1399
#> [ CPUFloatType{10,5} ]
```
