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
#>  0.0000  0.0000 -0.0233  0.0000  0.0845
#>  0.4847 -0.0771 -0.0000  0.0000  0.0000
#>  0.0000 -0.0000 -0.0000 -0.0000 -0.2520
#> -0.0000  1.0309 -0.0000  1.0780  0.0000
#>  0.2750 -0.0000 -0.0000  0.5290  0.5192
#> -2.3069 -1.0140  0.1098  0.0000  0.0000
#>  0.3420 -0.0000 -0.1091 -0.0000  0.6002
#> -0.1425 -0.0000  0.6772 -0.0000 -0.0000
#> -0.0089  0.0000  0.8630  0.0000  1.5284
#> -0.0482 -0.5337 -0.0000  0.0000 -0.1983
#> [ CPUFloatType{10,5} ]
```
