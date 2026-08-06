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
#> -0.0000 -0.0000 -0.2167 -0.0000  0.4053
#> -0.4437 -0.0000  4.2746 -0.3109  0.3918
#>  0.0000 -0.5031  0.0000 -0.3062  0.0208
#> -0.0000 -0.3381 -0.0193 -0.5060 -1.4728
#> -1.6814 -1.6636 -0.0510 -0.0000 -0.3714
#> -0.7938 -0.2297  0.0000  0.3010 -0.0137
#>  0.2940  0.0000  0.0000  0.0000 -0.0000
#>  0.0000  0.2647 -0.0000 -0.0540 -0.4740
#> -0.4154  0.2786  0.0000  0.1437 -0.5426
#> -0.0000 -0.1669 -0.0000  0.0000  0.0930
#> [ CPUFloatType{10,5} ]
```
