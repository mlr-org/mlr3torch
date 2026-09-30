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
#>  0.5981 -2.0682  0.0000  0.0000 -1.0187
#> -0.2531 -0.4137 -0.2830  0.2557 -0.0000
#> -0.5437 -0.4599  0.0000  0.0000  0.9734
#> -0.0000 -0.0000  0.0000  0.0000  0.1680
#> -0.0000 -0.0000  0.8869 -0.0000 -0.0000
#>  2.4058  1.1427 -0.0000 -0.0000 -0.3835
#>  0.5736  0.7321  1.6544  0.0000 -0.4601
#> -0.5930  0.0000  0.0000 -0.0000 -0.0000
#>  0.0000 -0.0000 -0.1356 -0.2946  1.3310
#>  0.0000  1.0182  0.0000 -0.0000 -0.2176
#> [ CPUFloatType{10,5} ]
```
