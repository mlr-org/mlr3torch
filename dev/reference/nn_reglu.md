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
#>  0.2345  0.0000 -1.8095  0.0000  1.3468
#>  0.2004 -0.0000 -0.0000  0.0631  0.0664
#> -1.0943  0.4752  0.7748 -0.0000 -0.0000
#> -0.4139 -0.0000  0.0000  1.8910 -0.0000
#>  0.0000 -0.0126  0.0773  0.0000 -0.0000
#> -0.0000  0.1936  1.3069 -1.0181 -0.0777
#>  0.0881  0.3186 -0.0000  0.0000 -0.5871
#>  0.5471 -0.0955 -0.0000  0.0000  0.1107
#>  0.7007 -2.4208 -2.7192  0.8213 -0.0000
#> -0.0000  0.1943  0.0000 -0.7458  0.1595
#> [ CPUFloatType{10,5} ]
```
