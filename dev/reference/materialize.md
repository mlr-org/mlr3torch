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
#>  1.1372  1.2498 -0.9501
#>  1.9855 -2.2965  1.0053
#>  0.0850 -0.3670  0.1590
#>  1.3272  0.3526  1.0400
#> -0.7921 -0.1978  1.5910
#>  1.2701  0.6566 -0.5677
#>  0.6309 -0.2034 -1.6032
#>  0.4593  2.1151  0.4783
#>  0.6748  1.2951 -0.2173
#> -1.3856  0.4090 -0.2657
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.1372
#>  1.2498
#> -0.9501
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.9855
#> -2.2965
#>  1.0053
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.0850
#> -0.3670
#>  0.1590
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  1.3272
#>  0.3526
#>  1.0400
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.7921
#> -0.1978
#>  1.5910
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.2701
#>  0.6566
#> -0.5677
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.6309
#> -0.2034
#> -1.6032
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.4593
#>  2.1151
#>  0.4783
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.6748
#>  1.2951
#> -0.2173
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.3856
#>  0.4090
#> -0.2657
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.1372  1.2498 -0.9501
#>  1.9855 -2.2965  1.0053
#>  0.0850 -0.3670  0.1590
#>  1.3272  0.3526  1.0400
#> -0.7921 -0.1978  1.5910
#>  1.2701  0.6566 -0.5677
#>  0.6309 -0.2034 -1.6032
#>  0.4593  2.1151  0.4783
#>  0.6748  1.2951 -0.2173
#> -1.3856  0.4090 -0.2657
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.0532  0.6357  1.7645 -0.3685
#>  0.3429  0.0499  0.9666  0.8974
#>  0.7634  0.9612  0.8671 -0.6222
#> -0.4913 -0.8603 -2.0565 -0.2924
#> -1.1349 -0.1697  0.7401 -1.2260
#>  0.7311  0.4826 -0.9362 -1.5322
#>  1.0422  0.6512 -0.9542 -1.3260
#> -0.7130 -0.7686  1.7503 -0.9063
#> -1.6499  1.7606  0.0309 -1.8432
#>  0.7748 -0.5360 -1.5213 -1.7350
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.1372
#>  1.2498
#> -0.9501
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.9855
#> -2.2965
#>  1.0053
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.0850
#> -0.3670
#>  0.1590
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  1.3272
#>  0.3526
#>  1.0400
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.7921
#> -0.1978
#>  1.5910
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.2701
#>  0.6566
#> -0.5677
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.6309
#> -0.2034
#> -1.6032
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.4593
#>  2.1151
#>  0.4783
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.6748
#>  1.2951
#> -0.2173
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.3856
#>  0.4090
#> -0.2657
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.0532
#>  0.6357
#>  1.7645
#> -0.3685
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.3429
#>  0.0499
#>  0.9666
#>  0.8974
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.7634
#>  0.9612
#>  0.8671
#> -0.6222
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.4913
#> -0.8603
#> -2.0565
#> -0.2924
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.1349
#> -0.1697
#>  0.7401
#> -1.2260
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.7311
#>  0.4826
#> -0.9362
#> -1.5322
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  1.0422
#>  0.6512
#> -0.9542
#> -1.3260
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.7130
#> -0.7686
#>  1.7503
#> -0.9063
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.6499
#>  1.7606
#>  0.0309
#> -1.8432
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.7748
#> -0.5360
#> -1.5213
#> -1.7350
#> [ CPUFloatType{4} ]
#> 
#> 
```
