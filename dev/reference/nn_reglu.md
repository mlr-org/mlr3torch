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
#>  1.1785 -0.8851 -0.0000 -0.0144  0.0000
#>  0.0000  0.1828 -0.5575 -0.0000 -0.0000
#>  0.1641  0.3047 -0.0000  0.1612  0.0071
#> -0.2189  0.0000  1.1292  0.0000  0.0000
#> -0.0000 -0.0000  0.6660 -3.5334 -0.1889
#> -0.0000  0.0000  0.0669  0.0073 -0.2398
#>  0.5193 -0.0000 -0.4995  0.0000  0.0000
#>  0.0000 -0.0000 -0.6034  0.0000  0.0000
#> -0.0000 -0.0254  0.0978  0.0000  0.0000
#> -0.0000 -2.6906  0.0000 -1.1201 -0.0000
#> [ CPUFloatType{10,5} ]
```
