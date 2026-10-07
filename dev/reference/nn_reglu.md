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
#> -3.1304 -1.9212 -0.0000 -0.0661  0.7842
#>  0.0000 -0.0000  0.0000  0.0000  2.1840
#>  0.0000 -0.0000  0.4430  0.0000 -0.0000
#> -0.0000 -0.0403 -0.1923 -1.7898 -0.0008
#> -0.5613 -0.0000 -0.0000  0.7686  0.0000
#>  1.2416  3.0375  0.0000 -2.8715 -0.0000
#> -1.2660  0.3685  0.2894  0.0000  0.0000
#> -0.5838 -0.0730  0.0000  0.4879  0.0000
#>  0.0000  0.0000 -0.1145 -0.0000  0.0333
#>  0.0000 -0.8037  0.0000  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
