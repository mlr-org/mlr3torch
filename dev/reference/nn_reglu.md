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
#>  0.0000  0.0000 -0.0000 -0.0000 -3.6624
#>  1.5011  0.0000 -3.7284 -0.0000  0.2551
#>  0.7948 -0.0000 -0.0000  0.6050 -0.0000
#> -0.0000  0.0000 -0.0314  0.1104 -1.2513
#> -0.1229 -0.2357  0.0000  1.4606  1.7090
#> -0.0000  0.4766 -0.0037  0.1479 -0.0000
#>  0.0028 -0.0308  0.0000  0.0000  0.0000
#>  0.0000  0.0552  0.0000 -0.0000 -0.4547
#> -0.3939  0.0000 -0.0000 -0.0000  0.0000
#> -0.0895 -0.4670 -0.0000 -0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
