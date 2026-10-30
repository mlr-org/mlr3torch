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
#> -0.0000 -1.0351  1.2860 -0.7531 -0.0000
#> -0.0000 -0.9626 -0.0000  0.0000 -0.0000
#> -3.3376  0.0000  1.5001  0.7385  0.2127
#> -0.0661 -0.0000 -0.5198  0.4113  0.1576
#> -0.0000  0.0392 -1.3557 -0.0000  0.0554
#> -0.5935 -0.0000  0.0000 -0.2096 -0.0000
#> -1.3193  0.0000  0.6876  0.0000  0.0000
#>  0.0000 -0.0000 -0.0000 -0.0456 -0.0000
#> -0.1674  0.0374  0.5230  0.0000 -0.0000
#> -0.0000 -0.0713 -0.3054  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
