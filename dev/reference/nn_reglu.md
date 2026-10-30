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
#>  0.4944  0.0000  0.0000 -0.0000  0.0000
#>  0.9362 -0.4965 -0.4870 -2.2303 -0.0000
#>  0.0121  0.0000 -0.0000  0.0000  0.4244
#> -0.3656 -0.6524 -0.0000  0.0000  0.0000
#> -0.0000 -0.0000 -0.0000  3.5314 -0.9410
#>  0.7184  0.0000  6.1769 -0.0000  0.5087
#>  0.0000 -0.0000 -0.1342  1.5050 -0.2548
#> -0.0160 -0.0000  0.9843 -1.0523  0.0000
#>  0.3803 -0.0000  0.0000  0.0000 -0.0000
#> -0.1115  0.2126  0.0000  0.0000  0.0773
#> [ CPUFloatType{10,5} ]
```
