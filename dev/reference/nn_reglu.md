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
#>  0.3530  0.1656 -0.0000  0.1713 -0.0000
#>  0.0000 -0.2473 -0.8399 -2.1345  0.3894
#>  0.0000  0.0000  0.0000 -0.0000 -0.0000
#>  0.0000 -0.5123  0.0301 -0.0000  0.2595
#>  0.0000 -0.0693 -0.0000 -0.0210  0.0000
#> -0.6540 -1.6291  0.0000  0.2539 -0.0542
#>  0.0911  0.0000 -0.3835 -0.0000  0.0000
#>  0.0000 -0.0000  0.0086 -0.0000  0.0000
#>  0.1223 -0.0000  0.8556 -0.0000 -0.0000
#> -0.0000  1.4089  2.4053  0.0000  0.2312
#> [ CPUFloatType{10,5} ]
```
