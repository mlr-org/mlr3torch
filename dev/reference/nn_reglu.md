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
#>  0.0000  0.3378  1.6162 -0.0000  0.0000
#>  0.0500  0.6207 -0.0000  0.0000 -0.0000
#> -0.0000  0.0057  0.5491  0.0000  0.0000
#> -0.0000 -0.0000 -0.0000  0.7600 -0.0000
#>  0.0000  0.0000 -0.0000 -0.6888  2.5123
#> -0.1808 -0.0099 -0.0000 -0.0000 -0.1538
#>  0.6174 -0.0000 -0.0000  0.0000 -0.0000
#> -0.0508  0.0000 -0.2755 -0.0000 -0.4926
#>  0.1150  0.7863  0.0000  1.8575 -0.4247
#>  0.0000 -0.2840 -0.0000  0.2769 -0.0000
#> [ CPUFloatType{10,5} ]
```
