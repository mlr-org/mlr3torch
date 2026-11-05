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
#> -1.1083  0.2686 -0.2562  0.0000 -0.0000
#>  0.0000  0.2542  0.0000  0.0000  0.0000
#>  0.0000 -0.5183 -0.0000 -0.0000  0.1470
#>  0.5426 -0.0000  0.0000 -0.1044  0.0000
#> -0.1151 -0.0000  1.1994 -0.0000 -0.6132
#>  0.0000  0.0538 -0.0000 -0.0000  0.0000
#>  0.0000 -0.0541  0.2097  0.0185 -0.0001
#> -0.3010 -0.4071  0.0000 -0.0000  1.4063
#>  0.0574  0.1392  0.1470  1.5852 -0.0192
#>  0.1821  0.0000  0.0000  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
