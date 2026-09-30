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
#>  0.0000 -0.0000  0.0861  0.0000 -0.0915
#>  0.7114  0.9651 -0.0000 -1.3464 -0.0000
#> -0.0000  0.0000  0.0024  0.0000  0.0000
#> -0.0000  0.0000 -0.0000  0.0000  0.3074
#>  0.0567  1.3044 -0.0000  0.0000 -0.0000
#>  0.0000 -0.0000 -1.5840 -0.0000 -0.0000
#> -0.2940 -0.0358  0.7757 -0.3183  0.6923
#> -0.0000 -1.4624  2.5728 -0.0000 -0.0000
#>  0.0140 -0.0000  0.0000  0.0517 -0.0000
#>  0.3216 -0.0000  0.0000 -0.0000 -0.1132
#> [ CPUFloatType{10,5} ]
```
