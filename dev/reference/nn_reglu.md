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
#> -0.0000 -0.0000 -0.0000  0.0000  0.3047
#> -0.5911  0.0000  0.0000 -0.5363  0.0000
#>  2.8475  0.0728 -0.5363  2.0469  0.0258
#> -0.0000 -0.1976 -1.1861 -0.0000 -0.0000
#> -0.2268  1.3291  0.0000 -0.0000  0.0000
#> -0.0000 -2.5289 -0.0000  0.0000 -0.0053
#> -0.0000 -0.0000 -0.1596 -0.0000  0.0000
#>  0.0000 -0.0976  0.3481  0.0000  0.0319
#> -0.0000 -0.5232  0.0000  0.0314  0.2160
#> -0.0000 -0.8520  0.4401  0.7419 -0.0000
#> [ CPUFloatType{10,5} ]
```
