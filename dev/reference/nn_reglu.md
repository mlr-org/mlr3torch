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
#>  0.0000  0.0000 -0.0006  0.2283  0.0000
#> -0.2994 -0.1214  0.2266 -0.0022 -0.9202
#>  0.2841  0.2436 -0.0000 -1.7338  0.0000
#>  0.8957  0.0000  0.0000 -0.1708 -0.0000
#> -0.0000  0.0608 -1.5466  0.1266  0.3999
#>  0.0000  0.0049  2.5135  0.3596 -0.0000
#> -0.1038 -0.3261 -0.0000 -0.0000 -0.3561
#> -0.0000  0.1151 -1.9652  0.0000 -0.0000
#> -0.2536 -0.0839 -0.0816  0.0000 -0.5599
#>  0.0000  1.4671 -0.4669  0.2313 -1.1577
#> [ CPUFloatType{10,5} ]
```
