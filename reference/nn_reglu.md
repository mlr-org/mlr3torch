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
#>  0.0000 -0.0000 -0.0000 -1.9433  0.0000
#> -0.0000  0.0000 -0.0000  0.1785 -0.0000
#> -0.0054 -0.0000 -0.0000 -0.0000  0.0000
#> -0.3651  0.1287 -0.0000 -1.7719 -0.0000
#>  0.0000 -1.3523 -0.4808 -0.0000  2.2111
#>  0.0000 -0.0000  0.0000  0.9441  0.4403
#> -0.1705  0.0000  0.0000  0.0000 -0.8414
#>  0.7199  0.5990  0.0000  0.5506  0.5197
#>  0.8584 -1.1205  0.1636 -0.0450 -0.8164
#>  0.0006  0.2224  0.0618  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
