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
#>  0.2487  0.0000 -0.9580  0.0000  0.6116
#> -0.2011 -1.2781  0.0057  0.5966  0.0000
#> -0.0721  1.3047 -0.6931 -0.6254  3.0249
#> -0.0000 -0.0102 -0.0000 -0.0000 -0.2988
#>  0.0000  0.0000  0.1318  2.3670 -0.0000
#>  0.5375 -0.4694  0.0000  0.2964  0.0000
#> -0.4628 -0.0000  0.0000  0.0941  0.3910
#>  0.0000 -0.0000 -0.3948  0.0000  0.0000
#>  0.0000 -0.2083 -0.0000  0.0000  0.0000
#>  0.0000 -0.3794 -1.1632  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
