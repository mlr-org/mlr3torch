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
#>  0.1463 -1.0413  0.0000  0.0000 -0.0000
#> -0.0000  0.0000 -0.0000 -0.0027  0.0000
#>  0.0000  0.1954 -0.1027  0.0000 -0.3873
#> -0.2639 -0.0000  0.2489 -0.0000  0.7289
#> -0.4802  0.0381 -0.0713 -0.0010  1.5605
#> -0.1663  0.1927 -0.0000 -0.5036 -0.0000
#> -0.0000 -3.6110 -0.0000  0.0000 -0.0000
#> -0.0000  0.0000  0.0000 -2.0104 -0.0000
#> -0.0000 -1.7756 -3.7634 -0.1600 -0.0985
#> -0.0124  0.0000 -0.0178  1.0043  0.0000
#> [ CPUFloatType{10,5} ]
```
