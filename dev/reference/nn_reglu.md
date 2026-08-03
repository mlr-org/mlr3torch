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
#> -0.0000 -1.0841  0.0000 -0.6202 -1.6533
#>  0.0000 -0.8644  0.0000 -0.0000  0.0000
#> -0.0000 -0.0000  0.0000 -0.0122 -0.0000
#> -0.3961  0.0062 -0.2300 -0.0000  0.0000
#>  0.7368  3.7616 -0.0000  0.5978  0.5582
#>  0.0000 -0.3517 -1.1464 -0.0000 -0.4306
#>  0.0000  0.0000 -0.0000 -0.3781 -0.8567
#>  0.0297  0.3835 -0.0000 -0.0000  0.7932
#> -1.2126  0.0000 -0.1358 -0.0000 -1.9846
#>  0.0000  0.0000  0.2675  0.8648 -0.0622
#> [ CPUFloatType{10,5} ]
```
