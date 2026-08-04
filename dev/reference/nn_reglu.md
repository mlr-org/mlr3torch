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
#>  0.3466  0.0296 -0.2922  0.0000 -0.5506
#>  0.0000 -0.5773 -4.2358  0.2957 -0.0209
#> -0.0000  0.0000  0.0000  0.0000 -1.1816
#> -0.4163 -0.3200 -0.0000  0.0000  0.0000
#> -0.0000 -0.0000 -2.1955  1.9357  1.4305
#> -0.3842  1.9779  0.0000  0.1932  0.0000
#>  0.0000  0.1049 -0.3889 -0.0000 -0.0000
#>  0.0000 -0.0000 -0.0000  1.5690 -0.7995
#>  1.4857 -0.0703 -0.4745 -0.0000 -0.0000
#>  0.0000 -0.0000  0.0000  0.0125 -0.0170
#> [ CPUFloatType{10,5} ]
```
