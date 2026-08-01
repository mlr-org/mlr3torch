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
#> -0.3170  0.0006  0.0041 -0.0000 -0.1642
#>  0.0000 -0.0000 -0.0000 -0.0000 -0.0000
#>  0.0000  0.0000  0.0942 -0.4136  0.0000
#>  0.4205  0.0000  0.0000 -1.9672  1.8997
#> -0.0000  0.0000 -2.2442  0.0000 -0.0000
#>  0.0000 -0.0227 -1.8261 -0.0000  0.2595
#> -0.0000  0.0000 -0.4344 -0.0000  0.0195
#> -0.2109 -1.4908  0.0000 -0.0000 -0.5453
#> -0.0000 -0.2935 -0.0000  0.0829  0.0000
#>  0.1387  0.0000 -2.1313  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
