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
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)s
or a
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html))

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
#> -0.6802  0.3399  0.3037
#> -0.3645 -0.3795  1.9631
#> -0.7864  0.3489 -1.2574
#> -0.5078  3.7562 -0.9980
#> -0.2337  0.4750  0.5490
#> -0.5676 -0.4248  0.6249
#> -0.3891  0.2184  1.0682
#> -0.7675  0.8410 -1.1928
#>  0.8787  0.0404 -1.2386
#>  0.2084 -0.9127  2.1875
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.6802
#>  0.3399
#>  0.3037
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3645
#> -0.3795
#>  1.9631
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.7864
#>  0.3489
#> -1.2574
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.5078
#>  3.7562
#> -0.9980
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2337
#>  0.4750
#>  0.5490
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.5676
#> -0.4248
#>  0.6249
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.3891
#>  0.2184
#>  1.0682
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.7675
#>  0.8410
#> -1.1928
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.8787
#>  0.0404
#> -1.2386
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.2084
#> -0.9127
#>  2.1875
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.6802  0.3399  0.3037
#> -0.3645 -0.3795  1.9631
#> -0.7864  0.3489 -1.2574
#> -0.5078  3.7562 -0.9980
#> -0.2337  0.4750  0.5490
#> -0.5676 -0.4248  0.6249
#> -0.3891  0.2184  1.0682
#> -0.7675  0.8410 -1.1928
#>  0.8787  0.0404 -1.2386
#>  0.2084 -0.9127  2.1875
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2455  0.6093  0.9931 -1.1719
#>  0.5089  1.3268  0.2172  0.7838
#>  0.8578 -0.0843  2.0002 -2.4179
#> -0.7151 -0.1490  0.4956  0.7396
#> -0.2309 -1.2658  0.5938 -1.4503
#> -0.8539  0.7734 -0.3023  0.7476
#> -0.8695  0.1921  0.9024  0.3853
#>  1.0573 -0.3783  2.2134  1.7052
#>  1.0349  0.3705 -0.1900  0.4674
#>  0.0025  0.1863  0.6212 -0.7798
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.6802
#>  0.3399
#>  0.3037
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3645
#> -0.3795
#>  1.9631
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.7864
#>  0.3489
#> -1.2574
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.5078
#>  3.7562
#> -0.9980
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2337
#>  0.4750
#>  0.5490
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.5676
#> -0.4248
#>  0.6249
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.3891
#>  0.2184
#>  1.0682
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.7675
#>  0.8410
#> -1.1928
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.8787
#>  0.0404
#> -1.2386
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.2084
#> -0.9127
#>  2.1875
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2455
#>  0.6093
#>  0.9931
#> -1.1719
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.5089
#>  1.3268
#>  0.2172
#>  0.7838
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.8578
#> -0.0843
#>  2.0002
#> -2.4179
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.7151
#> -0.1490
#>  0.4956
#>  0.7396
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.2309
#> -1.2658
#>  0.5938
#> -1.4503
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.8539
#>  0.7734
#> -0.3023
#>  0.7476
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.8695
#>  0.1921
#>  0.9024
#>  0.3853
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.0573
#> -0.3783
#>  2.2134
#>  1.7052
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.0349
#>  0.3705
#> -0.1900
#>  0.4674
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.0025
#>  0.1863
#>  0.6212
#> -0.7798
#> [ CPUFloatType{4} ]
#> 
#> 
```
