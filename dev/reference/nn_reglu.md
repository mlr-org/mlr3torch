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
#> -0.4380 -0.0000 -0.4008  0.6623 -0.0000
#>  0.1680 -1.1895  0.0000 -0.9769  0.0381
#> -1.3716 -0.4923 -0.0000  0.0000  0.5305
#> -0.0000 -0.1072 -0.0000 -0.2542  1.0319
#>  0.0000  0.0851  0.3905  0.0000  0.0000
#>  0.0000  0.2716  0.1381  0.0000  0.1333
#> -0.0000 -0.0000  0.0000  0.0000 -0.0394
#> -0.0000  0.4126 -0.0000 -0.8225  0.0000
#> -0.0000 -1.1446 -0.0000  0.2833 -0.0000
#> -0.9346  0.0000 -0.7451  0.0000  0.4999
#> [ CPUFloatType{10,5} ]
```
