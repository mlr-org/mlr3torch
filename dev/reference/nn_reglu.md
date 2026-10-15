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
#> -0.2519  1.8969 -0.0000 -0.0000 -0.0000
#>  0.1200  0.0000 -0.0000 -0.0000  0.2677
#> -0.0000 -0.3439  0.0000 -0.4604  0.0000
#>  0.0000 -0.4590  0.0000 -0.1800  0.5103
#> -0.0045  0.0000  0.3438 -0.8925  0.6045
#> -0.0000 -0.4181  0.0000 -0.0000 -1.2614
#>  0.0000  0.0000  0.0000 -0.0000 -0.0000
#>  0.2082 -0.4420  0.0008  0.1829 -1.2938
#>  0.0000  0.0000  3.3040 -0.4602 -0.0000
#>  0.0219 -0.0000  0.0000  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
