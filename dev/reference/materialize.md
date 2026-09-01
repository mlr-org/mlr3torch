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

  (`character(1)` or [`hashtab()`](https://rdrr.io/r/utils/hashtab.html)
  or `NULL`)  
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

For this reason it is possible to provide a cache, which is a
[`hashtab()`](https://rdrr.io/r/utils/hashtab.html). The key for a) is
`list(dataset, indices)`, the key for b) is
`list(indices, dataset, graph, input_map)`. The dataset and the graph go
into the key as the objects themselves, so keys are compared with
[`identical()`](https://rdrr.io/r/base/identical.html) rather than being
digested into a string, and two different keys can never share an entry.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#> -1.6494 -0.1968 -0.0827
#>  0.7391 -1.3510 -1.1863
#>  0.1146 -1.4202  0.1539
#>  0.3833  1.0087 -0.0815
#>  0.9256 -0.3215 -0.6203
#> -2.4491  1.5977  0.3428
#> -1.0950 -0.0489 -0.1663
#>  1.6383  1.1194  1.3052
#>  0.8355  0.1010  1.0748
#>  0.3100  0.6378 -0.8984
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.6494
#> -0.1968
#> -0.0827
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.7391
#> -1.3510
#> -1.1863
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.1146
#> -1.4202
#>  0.1539
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.3833
#>  1.0087
#> -0.0815
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.9256
#> -0.3215
#> -0.6203
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -2.4491
#>  1.5977
#>  0.3428
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -1.0950
#> -0.0489
#> -0.1663
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.6383
#>  1.1194
#>  1.3052
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.8355
#>  0.1010
#>  1.0748
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.3100
#>  0.6378
#> -0.8984
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.6494 -0.1968 -0.0827
#>  0.7391 -1.3510 -1.1863
#>  0.1146 -1.4202  0.1539
#>  0.3833  1.0087 -0.0815
#>  0.9256 -0.3215 -0.6203
#> -2.4491  1.5977  0.3428
#> -1.0950 -0.0489 -0.1663
#>  1.6383  1.1194  1.3052
#>  0.8355  0.1010  1.0748
#>  0.3100  0.6378 -0.8984
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.8995 -0.3573  0.4279 -0.4193
#>  2.1157 -0.9411 -1.5264 -0.2831
#>  0.7082  0.3459  1.8454  0.3422
#> -0.9221 -1.5444 -1.3267  0.7706
#>  0.6652  0.1937  0.3764  3.0502
#> -1.2403 -0.3471 -1.3980  0.0674
#>  1.1281 -0.9692 -0.4343  0.0110
#>  0.2012 -0.7575  0.7944  2.2530
#> -0.1046 -1.6507 -0.2638  0.1449
#>  0.8258 -0.1920  0.9467 -2.1666
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.6494
#> -0.1968
#> -0.0827
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.7391
#> -1.3510
#> -1.1863
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.1146
#> -1.4202
#>  0.1539
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.3833
#>  1.0087
#> -0.0815
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.9256
#> -0.3215
#> -0.6203
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -2.4491
#>  1.5977
#>  0.3428
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -1.0950
#> -0.0489
#> -0.1663
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.6383
#>  1.1194
#>  1.3052
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.8355
#>  0.1010
#>  1.0748
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.3100
#>  0.6378
#> -0.8984
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.8995
#> -0.3573
#>  0.4279
#> -0.4193
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  2.1157
#> -0.9411
#> -1.5264
#> -0.2831
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.7082
#>  0.3459
#>  1.8454
#>  0.3422
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.9221
#> -1.5444
#> -1.3267
#>  0.7706
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.6652
#>  0.1937
#>  0.3764
#>  3.0502
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.2403
#> -0.3471
#> -1.3980
#>  0.0674
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  1.1281
#> -0.9692
#> -0.4343
#>  0.0110
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.2012
#> -0.7575
#>  0.7944
#>  2.2530
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.1046
#> -1.6507
#> -0.2638
#>  0.1449
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.8258
#> -0.1920
#>  0.9467
#> -2.1666
#> [ CPUFloatType{4} ]
#> 
#> 
```
