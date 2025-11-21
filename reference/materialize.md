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
#>  1.7571  1.2899  0.5155
#> -0.5854 -0.3631  1.4213
#> -1.5695 -1.7601  0.0609
#> -0.9395  1.1914  0.6737
#>  1.3003 -1.4362 -0.3063
#> -0.8417 -0.6562  0.2026
#> -0.2845  0.0890  0.6990
#> -0.8099  2.0651 -1.0758
#>  0.1692  0.5315 -0.7862
#>  1.8308  0.1294 -0.3693
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.7571
#>  1.2899
#>  0.5155
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.5854
#> -0.3631
#>  1.4213
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.5695
#> -1.7601
#>  0.0609
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.9395
#>  1.1914
#>  0.6737
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.3003
#> -1.4362
#> -0.3063
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.8417
#> -0.6562
#>  0.2026
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.2845
#>  0.0890
#>  0.6990
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.8099
#>  2.0651
#> -1.0758
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.1692
#>  0.5315
#> -0.7862
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.8308
#>  0.1294
#> -0.3693
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.7571  1.2899  0.5155
#> -0.5854 -0.3631  1.4213
#> -1.5695 -1.7601  0.0609
#> -0.9395  1.1914  0.6737
#>  1.3003 -1.4362 -0.3063
#> -0.8417 -0.6562  0.2026
#> -0.2845  0.0890  0.6990
#> -0.8099  2.0651 -1.0758
#>  0.1692  0.5315 -0.7862
#>  1.8308  0.1294 -0.3693
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.3204  1.3161 -2.6775  0.0124
#>  1.0341 -0.6156  0.2877  2.1930
#>  1.8169 -0.9007  0.9424  1.6531
#>  0.1683 -0.7890  1.8199  0.1350
#>  0.6812  0.6324  0.5741  0.6794
#>  0.6675  0.5519  1.4092 -0.7531
#> -0.8337 -0.0492  1.8759  0.2969
#> -1.5755  0.3680 -1.3095  0.6603
#>  0.3085  0.8999 -1.1087  1.6517
#>  0.5802  2.1284  0.5172 -1.0776
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.7571
#>  1.2899
#>  0.5155
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.5854
#> -0.3631
#>  1.4213
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.5695
#> -1.7601
#>  0.0609
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.9395
#>  1.1914
#>  0.6737
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.3003
#> -1.4362
#> -0.3063
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.8417
#> -0.6562
#>  0.2026
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.2845
#>  0.0890
#>  0.6990
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.8099
#>  2.0651
#> -1.0758
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.1692
#>  0.5315
#> -0.7862
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.8308
#>  0.1294
#> -0.3693
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.3204
#>  1.3161
#> -2.6775
#>  0.0124
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.0341
#> -0.6156
#>  0.2877
#>  2.1930
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.8169
#> -0.9007
#>  0.9424
#>  1.6531
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.1683
#> -0.7890
#>  1.8199
#>  0.1350
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.6812
#>  0.6324
#>  0.5741
#>  0.6794
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.6675
#>  0.5519
#>  1.4092
#> -0.7531
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.8337
#> -0.0492
#>  1.8759
#>  0.2969
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.5755
#>  0.3680
#> -1.3095
#>  0.6603
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.3085
#>  0.8999
#> -1.1087
#>  1.6517
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.5802
#>  2.1284
#>  0.5172
#> -1.0776
#> [ CPUFloatType{4} ]
#> 
#> 
```
