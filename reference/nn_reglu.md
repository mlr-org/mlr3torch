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
#>  0.0000 -1.1286 -0.0719 -0.0000  0.0000
#>  0.1872 -0.2149 -0.0000 -0.0544  0.4413
#> -0.1842 -0.0000 -0.0000  0.0784  0.0000
#> -0.0000  0.0034 -0.0000 -0.2999  0.8147
#> -0.0844 -0.1044  0.0000  0.0000  0.0000
#> -0.0000 -0.1850 -0.0000 -0.0000 -0.0000
#>  0.0000  0.0000  1.3549  0.3599  0.0000
#> -0.0454 -0.0159 -0.0032 -0.0639  0.0000
#>  0.0000  0.8923 -0.0885  1.2254  0.0000
#> -0.0000 -0.3026 -0.3788  2.3882  0.0000
#> [ CPUFloatType{10,5} ]
```
