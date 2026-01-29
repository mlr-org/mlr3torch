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
#>  0.0000 -0.0000  1.2175  0.1150  0.1370
#> -0.1283 -0.0000 -0.2730  0.7040  0.0000
#>  0.0000  1.3699 -0.0000  0.0000  0.0075
#> -0.0000  0.0909 -0.0872 -0.0000  0.4433
#>  0.7943 -0.0000  0.0000  0.0000 -0.1251
#>  0.3130  0.0000 -0.0000 -0.1955 -0.2762
#> -0.0000 -0.0000  0.0000 -0.8375 -0.0000
#> -0.0000 -0.1791  0.4324 -0.2818  0.8522
#>  0.1302  0.0000 -0.6893  0.1376  0.6272
#> -0.0000 -0.0000  0.0377  0.0000 -0.4052
#> [ CPUFloatType{10,5} ]
```
