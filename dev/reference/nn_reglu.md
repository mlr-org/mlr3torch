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
#> -0.5618  0.1260 -0.3961  0.0000 -0.1499
#> -0.0041 -0.9936  0.0000 -0.0000 -0.0000
#> -0.0000  0.0000 -0.0000 -0.0000  0.0596
#>  0.0000  1.3803 -0.2546 -0.0000 -0.0000
#>  0.0000 -0.0408  0.1504 -0.4403  0.0000
#> -0.1394  0.0000 -1.3520 -0.0000  0.0000
#> -0.2962  0.0000  0.0000  0.0390 -1.8304
#> -0.0000  1.6167  1.1404 -0.0000 -0.0781
#> -0.0000 -0.0000  1.1096  0.0000 -0.0000
#> -0.0000 -0.0000 -0.0000 -0.6879  0.3456
#> [ CPUFloatType{10,5} ]
```
