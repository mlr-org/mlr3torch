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
#>  1.0966  0.0543  0.8681  0.0000 -0.0000
#> -0.0000 -0.4088  0.0000 -0.4985 -0.0172
#> -0.0000 -0.0000  1.5062  0.0000  0.0000
#> -0.0000  0.0000  0.0000  0.3311  0.0000
#>  0.4341  0.0000 -0.0000 -0.5103 -1.5599
#>  0.0000  0.0000  0.0000 -0.0000 -0.0000
#>  0.1304 -0.0000  0.0392  0.0000  1.0173
#>  0.2819 -0.7430  0.4565  0.0000  0.0000
#> -0.9731 -0.0000 -0.0000  0.6266 -0.4874
#> -0.0000 -0.4729 -0.1238  0.0000 -0.4739
#> [ CPUFloatType{10,5} ]
```
