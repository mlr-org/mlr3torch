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
#>  0.0000 -0.1016 -0.0000  0.8913 -0.4321
#>  0.0000 -0.0326 -0.6249 -0.0000 -0.0916
#> -2.4026 -0.0000 -0.0000 -1.4794  0.0000
#>  0.0000 -3.4667  0.2255 -0.0000 -0.0000
#>  1.0034 -0.0000  0.0000  0.0000 -0.0000
#>  1.6837 -0.0000  0.0000  0.0000 -0.0000
#> -0.0964  0.0000 -0.5318 -0.0482 -0.0000
#>  0.0000 -0.0000  0.0291  0.0000 -0.0000
#> -1.8843  0.0000 -0.0000  0.3120  0.7293
#>  0.0000 -1.4447 -0.0000  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
