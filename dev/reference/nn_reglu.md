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
#>  0.0000  0.1088  0.0029  0.5502  0.0000
#>  0.7131  0.6708 -0.0547  0.0000  0.0000
#> -0.0038 -0.5098 -0.1792  0.0000  0.0000
#> -0.0000 -0.0000  0.0000 -0.0000  1.5919
#>  0.1270  0.0000  0.0000  0.0190 -0.0000
#>  0.1407  0.0000  0.0008 -0.0000 -0.0176
#>  0.0000  0.0000 -0.0000 -0.0000 -0.0000
#>  0.3071 -0.5209  0.0000  0.0577 -0.5161
#>  0.6679  0.0000 -0.0048  0.0000 -0.2356
#>  0.0000  0.0000 -0.0000 -0.0852  0.1374
#> [ CPUFloatType{10,5} ]
```
