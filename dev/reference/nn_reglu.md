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
#> -1.5797  0.1398 -0.6919  0.0000  0.2165
#> -0.0641  0.0000  0.0000 -0.0000 -0.0000
#>  0.0000 -0.0795  0.0335 -0.6652 -0.5117
#> -0.0000  3.1830  0.0000 -0.3058 -0.0521
#> -0.0000 -1.7068 -0.0000  0.0000 -0.0195
#>  0.0000  0.0000 -0.0000  0.0000  0.0000
#>  0.0000 -0.0000 -0.0000 -0.0000 -0.1274
#>  0.0043 -0.7761  0.0000 -0.0673 -0.0000
#> -0.0000  0.5024 -1.8623 -0.0000  0.3879
#> -0.0000 -0.0000 -0.5222  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
