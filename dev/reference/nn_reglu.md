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
#>  0.0000 -0.0000 -1.7608  0.0650  0.0000
#> -1.8907  0.0236  0.8754 -0.0000  0.0000
#>  0.1199 -0.0000  2.3176  0.0000  0.9547
#> -0.1230 -0.0000 -0.1207 -0.1816 -0.3159
#> -0.0000  0.5503 -0.3337  1.4094 -0.6767
#> -3.0119  1.7864  1.4960 -0.4878 -0.0752
#>  0.6907  0.0000 -0.0000 -0.8010 -0.0000
#>  0.0000 -0.0000  0.0000  0.0000  0.7833
#>  0.2374  2.6252 -0.0000  0.7786  0.0000
#>  0.0000 -0.0000 -0.3586 -0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
