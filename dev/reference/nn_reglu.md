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
#>  0.0062 -0.1607  0.2673 -0.0000 -0.0000
#>  0.0000  0.1975 -0.0000 -0.0000  0.0000
#>  0.0928  0.1830  0.1562  0.2660  0.0000
#>  0.0000  0.0000 -0.0000 -0.1215  0.5922
#>  0.0000  0.0000 -0.3986 -0.0000 -0.0000
#> -0.0000 -0.0000 -0.0000  0.0000 -1.4372
#> -0.0000 -0.1962  0.3202  0.0000 -0.0070
#> -0.0000  0.7811 -0.0000  0.0000  0.0938
#>  0.0000 -0.1518 -0.5640 -0.9303 -0.0000
#>  0.0000 -1.0708  0.0000 -0.7659 -0.0000
#> [ CPUFloatType{10,5} ]
```
