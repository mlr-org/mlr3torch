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
#>  0.5161 -0.2900 -0.1805
#> -0.8131 -0.4081  0.8629
#>  0.2635 -0.1835  2.0408
#>  0.1243  0.6044 -0.7610
#>  0.8546 -0.1204 -0.1723
#>  1.3971 -0.6957  0.3176
#>  1.1252  0.8893  1.4672
#> -1.5148 -1.2551 -1.6075
#>  0.6660 -1.5412 -0.5746
#>  0.7561  1.1338 -0.4626
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.5161
#> -0.2900
#> -0.1805
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.8131
#> -0.4081
#>  0.8629
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.2635
#> -0.1835
#>  2.0408
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.1243
#>  0.6044
#> -0.7610
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.8546
#> -0.1204
#> -0.1723
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.3971
#> -0.6957
#>  0.3176
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.1252
#>  0.8893
#>  1.4672
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.5148
#> -1.2551
#> -1.6075
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.6660
#> -1.5412
#> -0.5746
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.7561
#>  1.1338
#> -0.4626
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.5161 -0.2900 -0.1805
#> -0.8131 -0.4081  0.8629
#>  0.2635 -0.1835  2.0408
#>  0.1243  0.6044 -0.7610
#>  0.8546 -0.1204 -0.1723
#>  1.3971 -0.6957  0.3176
#>  1.1252  0.8893  1.4672
#> -1.5148 -1.2551 -1.6075
#>  0.6660 -1.5412 -0.5746
#>  0.7561  1.1338 -0.4626
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.1101 -0.6072  0.8030 -0.4156
#> -0.1222 -1.4226  0.7172  0.8717
#> -0.0044 -0.2501 -0.2860  0.8235
#> -2.5761  0.5364 -0.0065  1.6388
#>  1.8424  1.2278  1.1405  0.2876
#> -0.2988  1.3654  1.9439  0.6868
#> -0.3700  1.5724  1.2780 -1.2548
#>  1.1889  1.0712 -0.1140 -0.5294
#>  2.0286  0.4045 -0.3070  1.0249
#> -1.0739  1.5357  0.5347  0.7258
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.5161
#> -0.2900
#> -0.1805
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.8131
#> -0.4081
#>  0.8629
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.2635
#> -0.1835
#>  2.0408
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.1243
#>  0.6044
#> -0.7610
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.8546
#> -0.1204
#> -0.1723
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.3971
#> -0.6957
#>  0.3176
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.1252
#>  0.8893
#>  1.4672
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.5148
#> -1.2551
#> -1.6075
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.6660
#> -1.5412
#> -0.5746
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.7561
#>  1.1338
#> -0.4626
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.1101
#> -0.6072
#>  0.8030
#> -0.4156
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.1222
#> -1.4226
#>  0.7172
#>  0.8717
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.0044
#> -0.2501
#> -0.2860
#>  0.8235
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -2.5761
#>  0.5364
#> -0.0065
#>  1.6388
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.8424
#>  1.2278
#>  1.1405
#>  0.2876
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.2988
#>  1.3654
#>  1.9439
#>  0.6868
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.3700
#>  1.5724
#>  1.2780
#> -1.2548
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.1889
#>  1.0712
#> -0.1140
#> -0.5294
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  2.0286
#>  0.4045
#> -0.3070
#>  1.0249
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.0739
#>  1.5357
#>  0.5347
#>  0.7258
#> [ CPUFloatType{4} ]
#> 
#> 
```
