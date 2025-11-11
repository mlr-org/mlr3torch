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
#> -0.0000 -0.9444  0.0000  0.0000 -0.0000
#>  0.0000  0.0000  0.0000 -0.0000 -0.0000
#>  0.0213  0.0000 -0.6993  0.0000 -0.0000
#>  0.4207 -0.0125 -1.9406 -0.3333  0.0000
#> -0.0000  0.0464 -1.0134 -0.0055 -0.0000
#> -0.1293  3.6106  2.5585  0.0000 -0.0000
#> -0.2633 -0.0000 -0.1546  0.0000 -0.1961
#> -0.0000 -2.7745  0.0000  0.1828  7.1987
#>  0.0708  0.0000 -1.4529 -0.0301  0.0000
#> -0.0000  1.6843 -0.0000 -1.1627  0.0000
#> [ CPUFloatType{10,5} ]
```
