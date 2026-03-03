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
#> -1.2002 -2.4027  0.5757
#>  0.0619  0.5323  2.4415
#>  1.5554  1.1841  1.1284
#> -0.4214 -0.4090 -0.1162
#> -0.5286  0.1931 -0.8144
#> -0.8293 -0.0231 -0.3744
#> -2.8205  0.7885  0.9534
#>  0.7401  1.6598 -1.7170
#> -1.8316  1.2827  0.3232
#> -2.0515 -1.0765 -0.5393
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.2002
#> -2.4027
#>  0.5757
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.0619
#>  0.5323
#>  2.4415
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.5554
#>  1.1841
#>  1.1284
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.4214
#> -0.4090
#> -0.1162
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.5286
#>  0.1931
#> -0.8144
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.8293
#> -0.0231
#> -0.3744
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -2.8205
#>  0.7885
#>  0.9534
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.7401
#>  1.6598
#> -1.7170
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.8316
#>  1.2827
#>  0.3232
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -2.0515
#> -1.0765
#> -0.5393
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.2002 -2.4027  0.5757
#>  0.0619  0.5323  2.4415
#>  1.5554  1.1841  1.1284
#> -0.4214 -0.4090 -0.1162
#> -0.5286  0.1931 -0.8144
#> -0.8293 -0.0231 -0.3744
#> -2.8205  0.7885  0.9534
#>  0.7401  1.6598 -1.7170
#> -1.8316  1.2827  0.3232
#> -2.0515 -1.0765 -0.5393
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2800  0.0584 -1.0960 -0.1311
#> -0.5486  0.4113 -1.3444  0.3392
#>  0.0545 -0.1642 -0.2851  0.8190
#> -0.1405  0.0203 -0.6879 -0.3676
#>  0.4937  1.9200 -1.1143  0.1263
#> -0.2097  0.0070 -1.8565 -1.1809
#> -0.6362 -0.9141 -2.4178 -0.1964
#> -0.4217  0.4945 -0.4572  1.0452
#>  0.6382  1.4993 -1.0520 -0.3845
#> -0.7071  0.1082  0.3802  0.7833
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.2002
#> -2.4027
#>  0.5757
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.0619
#>  0.5323
#>  2.4415
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.5554
#>  1.1841
#>  1.1284
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.4214
#> -0.4090
#> -0.1162
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.5286
#>  0.1931
#> -0.8144
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.8293
#> -0.0231
#> -0.3744
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -2.8205
#>  0.7885
#>  0.9534
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.7401
#>  1.6598
#> -1.7170
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.8316
#>  1.2827
#>  0.3232
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -2.0515
#> -1.0765
#> -0.5393
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2800
#>  0.0584
#> -1.0960
#> -0.1311
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.5486
#>  0.4113
#> -1.3444
#>  0.3392
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.0545
#> -0.1642
#> -0.2851
#>  0.8190
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.1405
#>  0.0203
#> -0.6879
#> -0.3676
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.4937
#>  1.9200
#> -1.1143
#>  0.1263
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.2097
#>  0.0070
#> -1.8565
#> -1.1809
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.6362
#> -0.9141
#> -2.4178
#> -0.1964
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.4217
#>  0.4945
#> -0.4572
#>  1.0452
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.6382
#>  1.4993
#> -1.0520
#> -0.3845
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.7071
#>  0.1082
#>  0.3802
#>  0.7833
#> [ CPUFloatType{4} ]
#> 
#> 
```
