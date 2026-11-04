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
#>  0.2528 -0.0000  1.3578 -1.3981 -1.6472
#> -0.2830 -2.6173 -0.0000 -0.5344  0.1344
#>  0.0000  0.0000 -0.0290  0.7121 -0.8914
#>  0.0000 -0.0000  0.0000  0.0000  0.0000
#>  1.1918 -0.0382 -0.2996  0.0000  0.3542
#>  0.0000 -0.3131 -0.0000 -0.0000 -0.0000
#> -0.1058  0.0000 -0.0000  0.0000  0.0000
#>  0.0000  0.2482  1.9238 -0.0000  1.2209
#> -0.3475 -0.3292 -0.1036 -0.0000  0.0000
#> -1.9753  0.0687 -0.4790 -0.0646  0.6605
#> [ CPUFloatType{10,5} ]
```
