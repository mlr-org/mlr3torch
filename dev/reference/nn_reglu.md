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
#>  0.6454 -0.1589 -0.0000 -0.0242 -0.0000
#> -0.0000  0.2612 -2.5022 -0.9258  1.6247
#>  1.4743 -0.0000  0.0000 -0.1093  0.0000
#> -0.2530  0.2331 -0.6995 -1.1805 -0.2543
#> -0.5263 -0.4134 -0.0000 -0.9465 -1.5927
#> -0.4584 -0.0000  0.0501 -0.0000  0.0000
#>  0.3564  0.1295  0.0000 -1.7079 -0.1583
#>  0.0000 -0.0000  0.0000 -0.3603  0.0000
#>  0.7100  0.3822 -0.8757  0.4287  0.9412
#>  0.0000 -6.5931  0.0000  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
