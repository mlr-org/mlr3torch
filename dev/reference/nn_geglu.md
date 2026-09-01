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
#> -0.0705  0.0986  2.2480  0.0380 -0.1252
#> -1.2504 -1.8946  0.0282 -0.0786  0.1264
#>  0.1766 -0.2856  0.7613 -0.1770  0.0960
#> -0.1347  0.1071  0.1859 -3.6799 -0.0386
#>  0.0256 -0.1473  0.0525 -0.2013  1.6079
#> -1.8094  0.0642  0.0942  0.1909 -0.2407
#>  0.2077 -0.2396 -0.1164 -0.0643 -0.0196
#>  1.1463  0.8201 -0.0821 -0.0908  0.0822
#>  0.3137 -0.0737  0.1816  0.0161 -0.0390
#>  0.7722 -0.5144 -0.0828  0.0454  0.0875
#> [ CPUFloatType{10,5} ]
```
