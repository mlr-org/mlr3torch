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
#>  0.0000 -1.0492  1.0374 -0.5756 -0.0000
#>  0.0000  0.0000 -0.0000  0.0000  0.3648
#> -0.0000  0.3638 -0.0000  0.0000 -1.9416
#> -0.0590  0.0000 -0.0669 -0.0000 -0.6536
#> -0.3416 -0.0000  0.0492  0.0000 -0.0091
#> -0.0000 -0.0000 -0.0000 -0.1739  0.1762
#>  0.0000 -0.0000 -4.8861  0.0000 -0.0000
#> -0.0000 -0.0000 -0.2128 -0.0809 -0.0000
#> -0.0000 -2.0118 -0.0134  0.2807  0.8659
#> -0.0000 -0.4432 -0.5522 -0.2314  0.4453
#> [ CPUFloatType{10,5} ]
```
