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
#>  0.0256 -0.6406  0.0000  0.3129  1.1079
#> -0.1357 -0.8133  0.0000  0.0000  0.0000
#>  0.0000 -0.0000  0.0000 -0.3232 -0.0000
#> -0.0000 -0.4738 -0.0000  0.0000 -0.1117
#> -1.9958  1.2407 -3.2134  0.0800 -1.2957
#> -0.0000  1.1788  0.8814 -0.0000  0.3489
#> -0.0000  0.9739  0.0000  2.9930 -0.9698
#>  0.5488 -0.0000 -0.0269 -0.0143  0.0000
#> -0.0000  0.0000 -0.0000  0.0000  1.7193
#> -0.0000 -0.1339  0.4057  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
