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
#>  0.0000  0.0000 -0.0000  0.5103  0.0000
#>  0.0000  0.0000 -0.0000  0.0000 -2.2662
#>  0.0000  0.0000  0.1008  0.0000  0.0000
#>  0.0000  0.0683 -0.0000 -0.0000 -1.0518
#>  0.8175 -0.4242  0.1981  0.0000  0.8650
#>  0.0000  1.3902 -0.0000  0.0000 -0.5345
#>  0.0667 -0.0000  0.1967  0.0000  0.2266
#>  0.3571 -0.0000 -0.0000 -0.2514  0.0000
#>  0.0000 -0.0462 -0.4449 -0.0000 -0.0000
#>  0.0000  0.0000  0.0000  2.5513  0.0000
#> [ CPUFloatType{10,5} ]
```
