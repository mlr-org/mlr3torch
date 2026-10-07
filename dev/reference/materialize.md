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
#>  0.3390  0.6839 -0.7433
#>  0.4642  0.5114 -1.2624
#> -0.6241 -1.5264  0.2515
#>  0.2568  0.1479  0.5249
#> -0.3059 -0.3395 -0.8120
#>  1.7039  0.2273  1.2960
#>  0.7514  0.5690  0.2517
#>  1.1745 -0.5280  0.6048
#> -0.6990 -0.6302  1.0537
#>  1.6389  0.1216  1.1774
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3390
#>  0.6839
#> -0.7433
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.4642
#>  0.5114
#> -1.2624
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.6241
#> -1.5264
#>  0.2515
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.2568
#>  0.1479
#>  0.5249
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.3059
#> -0.3395
#> -0.8120
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.7039
#>  0.2273
#>  1.2960
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.7514
#>  0.5690
#>  0.2517
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.1745
#> -0.5280
#>  0.6048
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.6990
#> -0.6302
#>  1.0537
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.6389
#>  0.1216
#>  1.1774
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3390  0.6839 -0.7433
#>  0.4642  0.5114 -1.2624
#> -0.6241 -1.5264  0.2515
#>  0.2568  0.1479  0.5249
#> -0.3059 -0.3395 -0.8120
#>  1.7039  0.2273  1.2960
#>  0.7514  0.5690  0.2517
#>  1.1745 -0.5280  0.6048
#> -0.6990 -0.6302  1.0537
#>  1.6389  0.1216  1.1774
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.7109  0.7073  1.7116  0.9630
#>  1.3336  0.2908  1.0069  1.2381
#> -0.5647  0.5422  0.4718 -1.1572
#>  0.7249 -2.7644  0.1861 -0.0398
#> -0.3883  0.2079  1.5402  0.1717
#>  0.1272  0.0764 -1.0450  0.5738
#> -1.1499  0.7460  0.7752  0.2749
#>  0.1897 -0.2804  0.2343 -0.6013
#>  1.9648  0.5158 -1.6564  1.0105
#>  0.7651  1.3361  1.1937  0.5254
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3390
#>  0.6839
#> -0.7433
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.4642
#>  0.5114
#> -1.2624
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.6241
#> -1.5264
#>  0.2515
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.2568
#>  0.1479
#>  0.5249
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.3059
#> -0.3395
#> -0.8120
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.7039
#>  0.2273
#>  1.2960
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.7514
#>  0.5690
#>  0.2517
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.1745
#> -0.5280
#>  0.6048
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.6990
#> -0.6302
#>  1.0537
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.6389
#>  0.1216
#>  1.1774
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.7109
#>  0.7073
#>  1.7116
#>  0.9630
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.3336
#>  0.2908
#>  1.0069
#>  1.2381
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.5647
#>  0.5422
#>  0.4718
#> -1.1572
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.7249
#> -2.7644
#>  0.1861
#> -0.0398
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.3883
#>  0.2079
#>  1.5402
#>  0.1717
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.1272
#>  0.0764
#> -1.0450
#>  0.5738
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.1499
#>  0.7460
#>  0.7752
#>  0.2749
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.1897
#> -0.2804
#>  0.2343
#> -0.6013
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.9648
#>  0.5158
#> -1.6564
#>  1.0105
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.7651
#>  1.3361
#>  1.1937
#>  0.5254
#> [ CPUFloatType{4} ]
#> 
#> 
```
