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
#> -0.0387 -0.4240  0.0474  0.0229  0.1278
#>  0.2578  0.0763 -0.0739  0.0750 -0.6957
#>  0.0853 -0.0354 -0.0018  0.4311  0.5578
#> -0.0249  1.6492  0.0096  0.0777  0.1062
#> -0.1655  0.2499 -0.0724 -0.1260 -0.0434
#> -0.0507 -0.0844 -0.0221 -0.0561 -0.1333
#>  0.0623  0.7159 -0.1235 -0.0730  0.1755
#> -0.5472 -0.0606 -0.0595  0.0438 -1.5594
#> -0.1076 -0.0299 -0.2796 -0.0271 -0.0111
#>  0.0497 -0.0812  0.0017 -0.2041  0.3541
#> [ CPUFloatType{10,5} ]
```
