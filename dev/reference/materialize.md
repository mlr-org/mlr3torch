# Materialize Lazy Tensor Columns

This will materialize a
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
or a [`data.frame()`](https://rdrr.io/r/base/data.frame.html) /
[`list()`](https://rdrr.io/r/base/list.html) containing – among other
things –
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
columns. I.e. the data described in the underlying
[`DataDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/DataDescriptor.md)s
is loaded for the indices in the
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md),
is preprocessed and then put unto the specified device. Because not all
elements in a lazy tensor must have the same shape, a list of tensors is
returned by default. If all elements have the same shape, these tensors
can also be rbinded into a single tensor (parameter `rbind`).

## Usage

``` r
materialize(x, device = "cpu", rbind = FALSE, ...)

# S3 method for class 'list'
materialize(x, device = "cpu", rbind = FALSE, cache = "auto", ...)
```

## Arguments

- x:

  (any)  
  The object to materialize. Either a
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  or a [`list()`](https://rdrr.io/r/base/list.html) /
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html) containing
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  columns.

- device:

  (`character(1)`)  
  The torch device.

- rbind:

  (`logical(1)`)  
  Whether to rbind the lazy tensor columns (`TRUE`) or return them as a
  list of tensors (`FALSE`). In the second case, there is no batch
  dimension.

- ...:

  (any)  
  Additional arguments.

- cache:

  (`character(1)` or
  [`environment()`](https://rdrr.io/r/base/environment.html) or
  `NULL`)  
  Optional cache for (intermediate) materialization results. Per
  default, caching will be enabled when the same dataset or data
  descriptor (with different output pointer) is used for more than one
  lazy tensor column.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)s
or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md))

## Details

Materializing a lazy tensor consists of:

1.  Loading the data from the internal dataset of the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/DataDescriptor.md).

2.  Processing these batches in the preprocessing
    [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s.

3.  Returning the result of the
    [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
    pointed to by the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/DataDescriptor.md)
    (`pointer`).

With multiple
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
columns we can benefit from caching because: a) Output(s) from the
dataset might be input to multiple graphs. b) Different lazy tensors
might be outputs from the same graph.

For this reason it is possible to provide a cache environment. The hash
key for a) is the hash of the indices and the dataset. The hash key for
b) is the hash of the indices, dataset and preprocessing graph.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#> -1.0024 -0.4413 -1.0595
#>  0.4324 -1.6653  2.4829
#>  0.0870  0.4091  1.4211
#>  0.1830  2.2127 -1.4505
#> -0.4159 -0.8781 -1.2130
#>  0.3590 -0.4357  0.3820
#> -0.7052  0.3997 -1.2777
#> -1.2458 -1.2342  0.0684
#> -0.0817 -1.0405 -0.2403
#>  0.1578  0.7320 -0.1118
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.0024
#> -0.4413
#> -1.0595
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.4324
#> -1.6653
#>  2.4829
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.0870
#>  0.4091
#>  1.4211
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.1830
#>  2.2127
#> -1.4505
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.4159
#> -0.8781
#> -1.2130
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.3590
#> -0.4357
#>  0.3820
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.7052
#>  0.3997
#> -1.2777
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.2458
#> -1.2342
#>  0.0684
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.0817
#> -1.0405
#> -0.2403
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.1578
#>  0.7320
#> -0.1118
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.0024 -0.4413 -1.0595
#>  0.4324 -1.6653  2.4829
#>  0.0870  0.4091  1.4211
#>  0.1830  2.2127 -1.4505
#> -0.4159 -0.8781 -1.2130
#>  0.3590 -0.4357  0.3820
#> -0.7052  0.3997 -1.2777
#> -1.2458 -1.2342  0.0684
#> -0.0817 -1.0405 -0.2403
#>  0.1578  0.7320 -0.1118
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.7123  2.2560  1.7679 -1.0181
#> -0.8448  0.4758 -0.6350 -0.7129
#>  0.3277  0.5196 -0.2551 -0.4825
#>  0.6248 -0.9364 -0.7424  1.3294
#> -0.3074 -0.2631  1.2975  0.5964
#>  0.6742  0.9044 -0.5787  0.4115
#>  0.5472  0.4858  0.5328 -0.8084
#>  0.6162 -2.0595  0.2186 -0.5256
#> -1.1881  0.7517 -0.6100 -1.8410
#> -1.1579 -0.2144 -1.5185  0.2095
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.0024
#> -0.4413
#> -1.0595
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.4324
#> -1.6653
#>  2.4829
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.0870
#>  0.4091
#>  1.4211
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.1830
#>  2.2127
#> -1.4505
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.4159
#> -0.8781
#> -1.2130
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.3590
#> -0.4357
#>  0.3820
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.7052
#>  0.3997
#> -1.2777
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.2458
#> -1.2342
#>  0.0684
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.0817
#> -1.0405
#> -0.2403
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.1578
#>  0.7320
#> -0.1118
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.7123
#>  2.2560
#>  1.7679
#> -1.0181
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.8448
#>  0.4758
#> -0.6350
#> -0.7129
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.3277
#>  0.5196
#> -0.2551
#> -0.4825
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.6248
#> -0.9364
#> -0.7424
#>  1.3294
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.3074
#> -0.2631
#>  1.2975
#>  0.5964
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.6742
#>  0.9044
#> -0.5787
#>  0.4115
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.5472
#>  0.4858
#>  0.5328
#> -0.8084
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.6162
#> -2.0595
#>  0.2186
#> -0.5256
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.1881
#>  0.7517
#> -0.6100
#> -1.8410
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.1579
#> -0.2144
#> -1.5185
#>  0.2095
#> [ CPUFloatType{4} ]
#> 
#> 
```
