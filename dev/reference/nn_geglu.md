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
#> -0.0987  0.0549 -0.1265 -0.2848 -0.0471
#>  0.0021  0.0558  0.0034  0.1736 -0.0724
#> -0.0628  0.2831 -0.0326 -0.0553 -0.6492
#>  0.5565 -0.5917  0.0011 -0.0808  0.0292
#> -0.1952 -0.8278 -0.5302 -0.8823  0.0163
#>  1.0990  0.0829 -0.0853  0.0255 -0.6335
#>  0.0430  1.1069  0.0795  0.0913  0.0362
#>  0.2263 -0.0964 -0.0204 -0.0064 -0.0673
#> -0.0213 -0.0291 -0.5867  0.0269 -0.1280
#>  0.6176 -0.0978  0.0325 -0.2240 -0.0740
#> [ CPUFloatType{10,5} ]
```
