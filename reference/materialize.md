# Materialize Lazy Tensor Columns

This will materialize a
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
or a [`data.frame()`](https://rdrr.io/r/base/data.frame.html) /
[`list()`](https://rdrr.io/r/base/list.html) containing – among other
things –
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
columns. I.e. the data described in the underlying
[`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)s
is loaded for the indices in the
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md),
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
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
  or a [`list()`](https://rdrr.io/r/base/list.html) /
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html) containing
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)s
or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md))

## Details

Materializing a lazy tensor consists of:

1.  Loading the data from the internal dataset of the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md).

2.  Processing these batches in the preprocessing
    [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s.

3.  Returning the result of the
    [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
    pointed to by the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)
    (`pointer`).

With multiple
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
#> -0.3988 -0.2448 -0.5395
#> -0.3621  0.1279 -0.9883
#>  0.2996  0.4375 -1.2752
#> -3.2109  0.4774 -0.8816
#> -0.6176  2.4325 -1.5962
#>  1.7528  0.5882 -0.5754
#>  0.3786  0.0634 -2.1979
#>  0.1262  0.6863  0.6974
#> -0.9003  1.1859 -1.0326
#> -0.6158 -1.5814  1.0516
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.3988
#> -0.2448
#> -0.5395
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3621
#>  0.1279
#> -0.9883
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.2996
#>  0.4375
#> -1.2752
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -3.2109
#>  0.4774
#> -0.8816
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.6176
#>  2.4325
#> -1.5962
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.7528
#>  0.5882
#> -0.5754
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.3786
#>  0.0634
#> -2.1979
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.1262
#>  0.6863
#>  0.6974
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.9003
#>  1.1859
#> -1.0326
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.6158
#> -1.5814
#>  1.0516
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.3988 -0.2448 -0.5395
#> -0.3621  0.1279 -0.9883
#>  0.2996  0.4375 -1.2752
#> -3.2109  0.4774 -0.8816
#> -0.6176  2.4325 -1.5962
#>  1.7528  0.5882 -0.5754
#>  0.3786  0.0634 -2.1979
#>  0.1262  0.6863  0.6974
#> -0.9003  1.1859 -1.0326
#> -0.6158 -1.5814  1.0516
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.7541 -1.2885 -1.0474  0.1754
#> -0.1148  0.3310 -0.8280 -0.1735
#>  0.3490 -1.1890 -1.5072 -0.0255
#>  1.2612 -0.9227  0.4943  0.5641
#>  0.6825 -1.3702  0.0131 -1.4096
#>  1.6627  0.4729  0.7704  0.3263
#> -1.4088  0.4764  1.1485 -0.2320
#> -0.2904 -0.4707  1.5496 -0.2154
#>  1.9400  1.3950  0.9423 -0.2292
#>  1.1070  0.5474  0.1123 -2.6793
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.3988
#> -0.2448
#> -0.5395
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3621
#>  0.1279
#> -0.9883
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.2996
#>  0.4375
#> -1.2752
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -3.2109
#>  0.4774
#> -0.8816
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.6176
#>  2.4325
#> -1.5962
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.7528
#>  0.5882
#> -0.5754
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.3786
#>  0.0634
#> -2.1979
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.1262
#>  0.6863
#>  0.6974
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.9003
#>  1.1859
#> -1.0326
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.6158
#> -1.5814
#>  1.0516
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.7541
#> -1.2885
#> -1.0474
#>  0.1754
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.1148
#>  0.3310
#> -0.8280
#> -0.1735
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.3490
#> -1.1890
#> -1.5072
#> -0.0255
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.2612
#> -0.9227
#>  0.4943
#>  0.5641
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.6825
#> -1.3702
#>  0.0131
#> -1.4096
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.6627
#>  0.4729
#>  0.7704
#>  0.3263
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.4088
#>  0.4764
#>  1.1485
#> -0.2320
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.2904
#> -0.4707
#>  1.5496
#> -0.2154
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.9400
#>  1.3950
#>  0.9423
#> -0.2292
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.1070
#>  0.5474
#>  0.1123
#> -2.6793
#> [ CPUFloatType{4} ]
#> 
#> 
```
