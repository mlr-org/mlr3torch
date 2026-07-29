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
#>  0.0309 -0.0000  0.0011 -0.0000 -0.0000
#>  0.0000  0.0000 -0.2607 -0.0000  0.0000
#> -0.0000  2.6809  0.0000  2.0059  0.2363
#>  0.0000  2.0500  0.2948  0.0000 -0.1425
#>  1.3782 -0.0626 -0.0226  0.0000 -0.0000
#> -0.2431  1.2556  0.0839 -0.0000 -0.8726
#>  0.0000  0.0000  0.0000  1.5408 -0.0000
#>  0.0000 -1.4532 -0.0496 -0.6946 -0.0000
#> -0.0000  0.0809  0.0000 -0.0000  0.3657
#> -0.0000  0.0000 -0.6878  0.0000 -0.7127
#> [ CPUFloatType{10,5} ]
```
