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
#>  0.0362 -0.6744  0.0714 -0.5688  0.0143
#>  0.6475  0.0000  0.3848 -0.2580  0.0000
#>  0.0326 -0.5440 -0.0000  0.0000  0.0000
#>  1.3042  0.0910 -0.0000  1.9452 -0.0000
#>  0.0000 -0.0000 -1.2454  0.0000  1.8949
#>  0.0000  0.0430  0.7757  0.0428  0.6242
#> -0.0046 -0.0000  0.0000 -0.0000 -0.0000
#> -0.0000 -1.1233  0.0000 -0.0000 -0.0000
#>  0.7854 -0.0111 -0.0000  1.7773  0.0000
#> -0.0000 -1.2354 -0.0334 -0.0903 -0.1919
#> [ CPUFloatType{10,5} ]
```
