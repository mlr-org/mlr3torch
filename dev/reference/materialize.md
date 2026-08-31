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
#> -0.3318  0.2591 -1.3612
#>  0.5824 -0.1665  0.7536
#> -1.6892 -0.9514 -1.0682
#> -0.0737  2.0841 -0.4702
#>  0.4576 -0.0466  1.1732
#> -0.1994  0.9983 -0.4518
#> -2.6445 -0.0021 -0.0122
#>  2.2114  1.5212  0.3188
#>  1.3819  0.1529 -2.8441
#> -0.4690  0.5152 -3.1897
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.3318
#>  0.2591
#> -1.3612
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.5824
#> -0.1665
#>  0.7536
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.6892
#> -0.9514
#> -1.0682
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.0737
#>  2.0841
#> -0.4702
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.4576
#> -0.0466
#>  1.1732
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.1994
#>  0.9983
#> -0.4518
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -2.6445
#> -0.0021
#> -0.0122
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  2.2114
#>  1.5212
#>  0.3188
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.3819
#>  0.1529
#> -2.8441
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.4690
#>  0.5152
#> -3.1897
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.3318  0.2591 -1.3612
#>  0.5824 -0.1665  0.7536
#> -1.6892 -0.9514 -1.0682
#> -0.0737  2.0841 -0.4702
#>  0.4576 -0.0466  1.1732
#> -0.1994  0.9983 -0.4518
#> -2.6445 -0.0021 -0.0122
#>  2.2114  1.5212  0.3188
#>  1.3819  0.1529 -2.8441
#> -0.4690  0.5152 -3.1897
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.4035 -0.4560 -0.5237 -0.1218
#>  0.6331  0.9924  0.9741 -0.6954
#> -0.6798 -0.5804 -2.6032 -0.8285
#> -0.6310  0.8388 -0.8790  0.2875
#> -0.1426 -0.2871 -2.2025  0.1228
#> -0.1864  2.8390 -0.4311 -1.7133
#>  0.2247  1.4626 -0.7149  1.9458
#> -0.6678 -0.5993  0.7331  0.0425
#> -0.5813 -0.6489 -0.5156  0.0185
#> -0.2925 -0.4612  1.6407 -0.3977
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.3318
#>  0.2591
#> -1.3612
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.5824
#> -0.1665
#>  0.7536
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.6892
#> -0.9514
#> -1.0682
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.0737
#>  2.0841
#> -0.4702
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.4576
#> -0.0466
#>  1.1732
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.1994
#>  0.9983
#> -0.4518
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -2.6445
#> -0.0021
#> -0.0122
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  2.2114
#>  1.5212
#>  0.3188
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.3819
#>  0.1529
#> -2.8441
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.4690
#>  0.5152
#> -3.1897
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.4035
#> -0.4560
#> -0.5237
#> -0.1218
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.6331
#>  0.9924
#>  0.9741
#> -0.6954
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.6798
#> -0.5804
#> -2.6032
#> -0.8285
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.6310
#>  0.8388
#> -0.8790
#>  0.2875
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.1426
#> -0.2871
#> -2.2025
#>  0.1228
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1864
#>  2.8390
#> -0.4311
#> -1.7133
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.2247
#>  1.4626
#> -0.7149
#>  1.9458
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.6678
#> -0.5993
#>  0.7331
#>  0.0425
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.5813
#> -0.6489
#> -0.5156
#>  0.0185
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.2925
#> -0.4612
#>  1.6407
#> -0.3977
#> [ CPUFloatType{4} ]
#> 
#> 
```
