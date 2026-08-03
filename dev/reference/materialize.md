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
#> -0.5445  0.8440  0.0569
#>  0.0126  0.7241 -0.2480
#> -1.1699  0.1885  0.1038
#> -0.6634 -2.0525 -0.7001
#>  0.3392 -0.9890 -0.1083
#>  0.6582  0.7280 -0.9372
#> -0.0102 -0.0498 -1.3881
#> -1.6084 -0.6430  2.1336
#> -1.9175 -0.9847 -1.2656
#> -0.3220 -0.4103 -1.3361
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.5445
#>  0.8440
#>  0.0569
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.0126
#>  0.7241
#> -0.2480
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.1699
#>  0.1885
#>  0.1038
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6634
#> -2.0525
#> -0.7001
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.3392
#> -0.9890
#> -0.1083
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.6582
#>  0.7280
#> -0.9372
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.0102
#> -0.0498
#> -1.3881
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.6084
#> -0.6430
#>  2.1336
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.9175
#> -0.9847
#> -1.2656
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.3220
#> -0.4103
#> -1.3361
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.5445  0.8440  0.0569
#>  0.0126  0.7241 -0.2480
#> -1.1699  0.1885  0.1038
#> -0.6634 -2.0525 -0.7001
#>  0.3392 -0.9890 -0.1083
#>  0.6582  0.7280 -0.9372
#> -0.0102 -0.0498 -1.3881
#> -1.6084 -0.6430  2.1336
#> -1.9175 -0.9847 -1.2656
#> -0.3220 -0.4103 -1.3361
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.8875 -0.1876 -1.9920  0.1944
#>  1.0756 -2.2623 -1.1159 -0.1884
#> -0.3179  0.8958 -1.0567  0.1719
#>  0.6345 -1.2021  0.2482  2.0691
#>  1.0371  0.6090 -0.2151 -0.1079
#>  0.1202  0.5196  1.4040 -0.3798
#> -0.1243 -0.9261  0.6927 -0.2725
#> -0.9523 -0.8339  0.9007 -1.0369
#>  0.4370  0.5113  1.1119 -0.3289
#> -0.3964 -0.0213 -2.3782 -1.1767
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.5445
#>  0.8440
#>  0.0569
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.0126
#>  0.7241
#> -0.2480
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.1699
#>  0.1885
#>  0.1038
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6634
#> -2.0525
#> -0.7001
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.3392
#> -0.9890
#> -0.1083
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.6582
#>  0.7280
#> -0.9372
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.0102
#> -0.0498
#> -1.3881
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.6084
#> -0.6430
#>  2.1336
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.9175
#> -0.9847
#> -1.2656
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.3220
#> -0.4103
#> -1.3361
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.8875
#> -0.1876
#> -1.9920
#>  0.1944
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.0756
#> -2.2623
#> -1.1159
#> -0.1884
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.3179
#>  0.8958
#> -1.0567
#>  0.1719
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.6345
#> -1.2021
#>  0.2482
#>  2.0691
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.0371
#>  0.6090
#> -0.2151
#> -0.1079
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.1202
#>  0.5196
#>  1.4040
#> -0.3798
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.1243
#> -0.9261
#>  0.6927
#> -0.2725
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.9523
#> -0.8339
#>  0.9007
#> -1.0369
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.4370
#>  0.5113
#>  1.1119
#> -0.3289
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.3964
#> -0.0213
#> -2.3782
#> -1.1767
#> [ CPUFloatType{4} ]
#> 
#> 
```
