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
#> -0.0000 -1.8969  0.0000  0.0000 -0.4134
#>  0.0000 -0.0000  0.0000 -0.0000  0.0000
#>  0.9161  0.5626  0.0000 -0.0000 -0.0000
#>  0.0000  1.4288 -0.0000 -0.8783 -0.0000
#> -0.0000  0.0303  0.0000 -0.3890  0.0000
#> -0.5811 -1.7973  0.1777 -0.6450  0.5727
#>  0.0937 -0.2325  0.1914 -0.0000 -0.4463
#>  0.0000 -0.3216  0.0000 -0.5112 -0.1194
#>  0.0000 -0.0000  0.0151 -0.2857 -0.0000
#>  0.0000  0.0000  0.4038  0.0961 -0.0000
#> [ CPUFloatType{10,5} ]
```
