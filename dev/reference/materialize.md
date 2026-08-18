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
#>  1.0927 -0.6016  0.3590
#> -0.2339  0.6017  0.2439
#> -0.2428  1.2638 -0.8107
#> -0.4334  0.6083 -0.2149
#>  1.2797  0.6423  1.8809
#>  0.6292  1.3300 -0.1083
#>  1.2088 -0.1040 -1.5455
#>  1.2327 -1.6020 -1.3374
#> -1.0243  1.0193  1.1847
#>  0.0734 -1.8274  1.7824
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.0927
#> -0.6016
#>  0.3590
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.2339
#>  0.6017
#>  0.2439
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.2428
#>  1.2638
#> -0.8107
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.4334
#>  0.6083
#> -0.2149
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.2797
#>  0.6423
#>  1.8809
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.6292
#>  1.3300
#> -0.1083
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.2088
#> -0.1040
#> -1.5455
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.2327
#> -1.6020
#> -1.3374
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.0243
#>  1.0193
#>  1.1847
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.0734
#> -1.8274
#>  1.7824
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.0927 -0.6016  0.3590
#> -0.2339  0.6017  0.2439
#> -0.2428  1.2638 -0.8107
#> -0.4334  0.6083 -0.2149
#>  1.2797  0.6423  1.8809
#>  0.6292  1.3300 -0.1083
#>  1.2088 -0.1040 -1.5455
#>  1.2327 -1.6020 -1.3374
#> -1.0243  1.0193  1.1847
#>  0.0734 -1.8274  1.7824
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0381 -0.7909  0.2778 -0.6100
#>  0.6866 -1.9342  0.1260 -1.2019
#> -0.5482 -0.0104  0.1381 -0.6508
#> -0.5005  2.1800  0.3240  0.6184
#>  0.2414  0.1581  2.1921  0.9093
#> -0.8751 -0.3916  0.7704  1.3348
#> -1.3235  3.4230 -0.5416 -1.9551
#>  0.1207  0.5064 -0.0093 -0.3189
#>  1.4750  1.6773  0.5383 -1.0490
#> -0.4214  0.6499  1.3702 -1.1986
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.0927
#> -0.6016
#>  0.3590
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.2339
#>  0.6017
#>  0.2439
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.2428
#>  1.2638
#> -0.8107
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.4334
#>  0.6083
#> -0.2149
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.2797
#>  0.6423
#>  1.8809
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.6292
#>  1.3300
#> -0.1083
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.2088
#> -0.1040
#> -1.5455
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.2327
#> -1.6020
#> -1.3374
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.0243
#>  1.0193
#>  1.1847
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.0734
#> -1.8274
#>  1.7824
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0381
#> -0.7909
#>  0.2778
#> -0.6100
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.6866
#> -1.9342
#>  0.1260
#> -1.2019
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.5482
#> -0.0104
#>  0.1381
#> -0.6508
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.5005
#>  2.1800
#>  0.3240
#>  0.6184
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.2414
#>  0.1581
#>  2.1921
#>  0.9093
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.8751
#> -0.3916
#>  0.7704
#>  1.3348
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.3235
#>  3.4230
#> -0.5416
#> -1.9551
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.1207
#>  0.5064
#> -0.0093
#> -0.3189
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.4750
#>  1.6773
#>  0.5383
#> -1.0490
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.4214
#>  0.6499
#>  1.3702
#> -1.1986
#> [ CPUFloatType{4} ]
#> 
#> 
```
