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
#> -0.1460 -0.2629 -0.0129  0.6935 -0.1282
#>  0.0000 -1.0315 -0.6848  2.2359 -0.0000
#> -0.0357  0.0000  0.2434 -0.7967  0.0000
#>  0.1518 -0.0000 -0.0000 -0.5561 -0.0000
#>  1.2298 -0.0847  0.1306 -0.2750  0.1477
#> -0.3232 -0.0000 -1.2836 -0.0000 -0.0000
#> -0.0000  0.4248  0.0984  0.0000 -0.3392
#>  0.0000  0.2183 -1.3534  0.3498  0.0000
#>  0.2495 -2.7381  0.0922 -0.5757  0.0000
#>  0.0769 -0.6830  0.0000  0.1919  0.0000
#> [ CPUFloatType{10,5} ]
```
