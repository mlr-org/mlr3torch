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
#>  0.4310 -0.9344 -0.2248
#>  0.7420 -0.1992  0.3633
#> -0.9828  0.2907  0.7156
#> -1.4364  0.9012  0.0198
#>  0.6686  0.0761  1.1363
#>  0.0218  0.4295  0.4669
#> -1.8094  0.4451  1.5293
#>  0.9622  0.2536 -0.3191
#> -1.3317 -0.3153  0.9011
#> -0.1243  1.4538  0.5957
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.4310
#> -0.9344
#> -0.2248
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.7420
#> -0.1992
#>  0.3633
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.9828
#>  0.2907
#>  0.7156
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.4364
#>  0.9012
#>  0.0198
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.6686
#>  0.0761
#>  1.1363
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.0218
#>  0.4295
#>  0.4669
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -1.8094
#>  0.4451
#>  1.5293
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.9622
#>  0.2536
#> -0.3191
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.3317
#> -0.3153
#>  0.9011
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.1243
#>  1.4538
#>  0.5957
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.4310 -0.9344 -0.2248
#>  0.7420 -0.1992  0.3633
#> -0.9828  0.2907  0.7156
#> -1.4364  0.9012  0.0198
#>  0.6686  0.0761  1.1363
#>  0.0218  0.4295  0.4669
#> -1.8094  0.4451  1.5293
#>  0.9622  0.2536 -0.3191
#> -1.3317 -0.3153  0.9011
#> -0.1243  1.4538  0.5957
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.1317  0.5904  0.6458  1.2529
#> -0.9816 -1.9465 -0.1871  0.2028
#>  0.5328  0.1846 -0.4428 -0.3672
#> -1.3968 -0.0462  0.3858  0.5322
#>  1.5064 -0.5501  0.3603 -2.1283
#> -0.8614  0.7854 -3.9424  0.1616
#> -0.4718  0.2375 -0.8937 -0.5100
#> -1.3532 -1.2688 -1.1288  0.6044
#> -2.3706  0.4604 -0.4139 -3.4894
#> -1.0589  2.2506  0.7057 -1.1096
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.4310
#> -0.9344
#> -0.2248
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.7420
#> -0.1992
#>  0.3633
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.9828
#>  0.2907
#>  0.7156
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.4364
#>  0.9012
#>  0.0198
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.6686
#>  0.0761
#>  1.1363
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.0218
#>  0.4295
#>  0.4669
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -1.8094
#>  0.4451
#>  1.5293
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.9622
#>  0.2536
#> -0.3191
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.3317
#> -0.3153
#>  0.9011
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.1243
#>  1.4538
#>  0.5957
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.1317
#>  0.5904
#>  0.6458
#>  1.2529
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.9816
#> -1.9465
#> -0.1871
#>  0.2028
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5328
#>  0.1846
#> -0.4428
#> -0.3672
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.3968
#> -0.0462
#>  0.3858
#>  0.5322
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.5064
#> -0.5501
#>  0.3603
#> -2.1283
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.8614
#>  0.7854
#> -3.9424
#>  0.1616
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.4718
#>  0.2375
#> -0.8937
#> -0.5100
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.3532
#> -1.2688
#> -1.1288
#>  0.6044
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -2.3706
#>  0.4604
#> -0.4139
#> -3.4894
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.0589
#>  2.2506
#>  0.7057
#> -1.1096
#> [ CPUFloatType{4} ]
#> 
#> 
```
