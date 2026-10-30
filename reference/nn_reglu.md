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
#>  0.0000  0.0028  0.0000  1.0030  0.0000
#>  0.0000 -0.2125 -0.0210  0.0000 -0.0000
#> -0.0000  0.0000 -1.4021  0.0000  0.0000
#> -0.0000 -0.0000 -0.1010 -0.3306 -0.0000
#>  1.5251 -1.2116  0.0000  0.0838 -0.0000
#> -0.0000  0.0000 -0.0885  0.0364  0.1663
#> -0.0734  0.0000  0.0328  0.0000 -0.0000
#> -0.0000 -0.9855 -0.0000  0.0000 -2.4736
#> -0.0000 -0.0000 -0.2872 -0.0010  0.0000
#> -0.7169  0.0000  0.0000 -0.0325 -0.0000
#> [ CPUFloatType{10,5} ]
```
