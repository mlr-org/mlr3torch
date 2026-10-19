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
#>  0.0000  0.0000 -1.0821  1.2607  0.0000
#> -0.0000 -0.0000 -1.9797 -2.7938 -0.0000
#> -0.0000  0.4136  0.9823 -0.0000  0.1953
#> -1.2653 -0.0000 -0.0000  2.4157 -0.0000
#>  0.5980  0.1642 -0.5122 -0.0000 -1.7199
#>  0.0000  0.0079 -0.0000 -0.5002  0.2511
#> -1.0602  0.0000 -1.6326  0.0000  0.5660
#>  0.0000  0.1296 -0.0640  0.0000  0.0000
#>  0.1296 -1.6114 -0.1542  0.8673 -0.6284
#>  0.4270  0.0000 -0.0000  0.4383  0.0000
#> [ CPUFloatType{10,5} ]
```
