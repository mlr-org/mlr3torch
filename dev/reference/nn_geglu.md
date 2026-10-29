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
#> -0.0007 -0.1197  0.0718 -0.0248  0.5513
#>  0.0469  0.4189  0.0316  0.1144  0.0285
#> -0.0822  0.0350 -0.4598  0.0031 -0.1595
#>  0.1512 -0.0104 -0.2337  0.1365 -0.2197
#>  0.0073 -3.2319  0.0061  0.2146  0.0407
#> -0.1112 -1.4146 -0.0172  0.2701 -0.1212
#> -0.0526  0.0604 -0.0887  0.4191  0.0505
#>  0.2788  0.4689  0.0922 -0.0011 -0.1875
#> -0.1178  0.0051 -0.0282  0.0681  0.2677
#>  0.0005  0.1294  0.0033 -0.7169 -0.0146
#> [ CPUFloatType{10,5} ]
```
