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
#> -0.0000 -0.4771 -0.4232 -0.0617 -0.1755
#>  0.0046 -1.8799  0.0000  0.2804  2.0065
#> -0.6886 -0.0000  0.1746 -0.0000  0.2282
#> -0.3699  0.0000  0.7279 -0.0000  0.0000
#> -0.0000  0.5832 -0.3754 -0.0000  0.6497
#>  1.7299 -0.4395  0.8163  2.1300 -1.1063
#> -0.0000 -0.0000  0.0000  0.0000 -0.0000
#> -0.7560 -0.6597  0.6533 -0.0000 -0.0000
#>  0.0000 -0.0849 -0.0000 -0.0446  1.4913
#> -0.0937  0.0000 -0.0696  0.7688  0.0951
#> [ CPUFloatType{10,5} ]
```
