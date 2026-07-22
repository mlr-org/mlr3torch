# GeGLU Module

This module implements the Gaussian Error Linear Unit Gated Linear Unit
(GeGLU) activation function. It computes \\\text{GeGLU}(x, g) = x \cdot
\text{GELU}(g)\\ where \\x\\ and \\g\\ are created by splitting the
input tensor in half along the last dimension.

## Usage

``` r
nn_geglu()
```

## References

Shazeer N (2020). “GLU Variants Improve Transformer.” 2002.05202,
<https://arxiv.org/abs/2002.05202>.

## Examples

``` r
x = torch::torch_randn(10, 10)
glu = nn_geglu()
glu(x)
#> torch_tensor
#>  0.0655 -0.1977 -0.0773  0.9059 -0.0397
#> -0.7105  0.1089  0.2386  0.5304  0.0037
#>  0.0199 -0.0656  0.1551 -0.0671  0.1048
#>  0.3351 -0.0449 -0.1312  2.5690  0.7546
#> -1.3536  0.0667 -0.1453  1.4376 -0.7128
#> -0.3031 -0.0091  0.0005  0.1953  0.1406
#> -2.7691 -0.2059 -0.0196  0.0621  0.0682
#>  0.0208  0.1368 -0.7377 -0.1900 -0.2619
#>  0.2344 -0.0306  0.1594 -0.0165 -0.3855
#>  2.2632  2.5565  0.0615  0.0283 -0.0928
#> [ CPUFloatType{10,5} ]
```
