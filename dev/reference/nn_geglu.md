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
#> -0.0977 -0.0183 -0.1105  0.4296  0.6120
#>  0.3205 -0.8371 -0.0448 -0.1895 -0.9297
#>  0.7965 -0.1156  0.2478 -0.0401 -0.0350
#>  0.2665  0.0900  0.0156  1.4258 -0.0581
#> -0.2004 -0.0013  0.0194  0.1201  0.6056
#> -1.5146 -0.0151  0.0966 -0.0056 -0.0286
#> -0.3375 -0.0113 -0.0188  0.0348  0.0656
#> -0.1968 -0.0890  0.0002 -2.4414 -0.0440
#> -0.0521  0.0154 -0.2394  0.0018 -0.0022
#> -0.0543  0.0304 -0.0302 -0.2247  0.4380
#> [ CPUFloatType{10,5} ]
```
