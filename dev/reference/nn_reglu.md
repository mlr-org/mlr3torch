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
#>  0.0000 -0.0000  0.1300  0.3631 -0.1123
#>  0.0921  0.7921  0.8413 -0.2309 -0.2423
#>  0.0000 -0.9009 -0.0869  0.3427 -1.0230
#> -0.3501  0.0048  0.0000 -0.8748  0.0000
#> -0.0000  1.2125  0.6139 -0.0000 -0.0000
#> -0.2009  0.0000  0.0000 -0.0423 -0.0736
#> -0.0000 -0.9538  0.3924 -0.4196  0.1010
#> -0.0000 -0.0000 -1.5732  0.0000  0.0000
#>  0.1265  1.6936 -0.0000 -1.5279  0.3359
#>  0.5680 -0.0379 -0.2275  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
