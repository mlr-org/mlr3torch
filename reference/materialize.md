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
#>  1.6381  0.0067  0.3437
#> -0.3202 -0.5759  1.3909
#> -0.4927  1.6539  0.9273
#>  0.0112  0.5345 -0.1019
#> -0.5788 -0.9218  3.1175
#>  0.0739  0.1976 -0.2857
#>  0.3719  1.1318 -1.1404
#> -0.6545  0.3379  0.5246
#> -0.3378  0.2813 -1.2368
#> -2.6155  0.4711  0.9210
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.6381
#>  0.0067
#>  0.3437
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3202
#> -0.5759
#>  1.3909
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.4927
#>  1.6539
#>  0.9273
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.0112
#>  0.5345
#> -0.1019
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.5788
#> -0.9218
#>  3.1175
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.0739
#>  0.1976
#> -0.2857
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.3719
#>  1.1318
#> -1.1404
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.6545
#>  0.3379
#>  0.5246
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.3378
#>  0.2813
#> -1.2368
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -2.6155
#>  0.4711
#>  0.9210
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.6381  0.0067  0.3437
#> -0.3202 -0.5759  1.3909
#> -0.4927  1.6539  0.9273
#>  0.0112  0.5345 -0.1019
#> -0.5788 -0.9218  3.1175
#>  0.0739  0.1976 -0.2857
#>  0.3719  1.1318 -1.1404
#> -0.6545  0.3379  0.5246
#> -0.3378  0.2813 -1.2368
#> -2.6155  0.4711  0.9210
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.1392 -0.3969 -1.1511  0.7838
#>  0.4329 -0.6018  0.3231 -1.5728
#> -0.2722 -0.1084  0.9675  0.9575
#>  0.4448  1.6764  0.8257 -0.1623
#> -0.4323 -2.2368 -1.0482 -0.2904
#>  0.0305 -1.0959 -1.7702  0.5021
#> -2.2578  1.0203 -0.7784  0.0892
#>  0.0421 -0.1414  1.8772 -2.1335
#> -0.5646  0.0216  0.3588  0.8955
#>  0.2124  2.3240  1.9158 -1.4635
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.6381
#>  0.0067
#>  0.3437
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3202
#> -0.5759
#>  1.3909
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.4927
#>  1.6539
#>  0.9273
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.0112
#>  0.5345
#> -0.1019
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.5788
#> -0.9218
#>  3.1175
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.0739
#>  0.1976
#> -0.2857
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.3719
#>  1.1318
#> -1.1404
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.6545
#>  0.3379
#>  0.5246
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.3378
#>  0.2813
#> -1.2368
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -2.6155
#>  0.4711
#>  0.9210
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.1392
#> -0.3969
#> -1.1511
#>  0.7838
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.4329
#> -0.6018
#>  0.3231
#> -1.5728
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.2722
#> -0.1084
#>  0.9675
#>  0.9575
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.4448
#>  1.6764
#>  0.8257
#> -0.1623
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.4323
#> -2.2368
#> -1.0482
#> -0.2904
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.0305
#> -1.0959
#> -1.7702
#>  0.5021
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -2.2578
#>  1.0203
#> -0.7784
#>  0.0892
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.0421
#> -0.1414
#>  1.8772
#> -2.1335
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.5646
#>  0.0216
#>  0.3588
#>  0.8955
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.2124
#>  2.3240
#>  1.9158
#> -1.4635
#> [ CPUFloatType{4} ]
#> 
#> 
```
