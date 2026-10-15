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
#> -0.6610  0.1783  2.9860
#> -1.6098  1.1380  0.2154
#> -1.1705  0.8160  0.3299
#>  0.2730 -2.1429  2.1008
#> -0.4236 -1.4043  0.9649
#>  0.7149 -0.1239  0.1402
#> -0.0684 -0.8429  0.9603
#> -0.9608 -2.1126 -1.3324
#> -0.8603  0.5753 -0.8214
#> -1.7250  0.7079 -0.5922
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.6610
#>  0.1783
#>  2.9860
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.6098
#>  1.1380
#>  0.2154
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.1705
#>  0.8160
#>  0.3299
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.2730
#> -2.1429
#>  2.1008
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.4236
#> -1.4043
#>  0.9649
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.7149
#> -0.1239
#>  0.1402
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.0684
#> -0.8429
#>  0.9603
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.9608
#> -2.1126
#> -1.3324
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.8603
#>  0.5753
#> -0.8214
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.7250
#>  0.7079
#> -0.5922
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.6610  0.1783  2.9860
#> -1.6098  1.1380  0.2154
#> -1.1705  0.8160  0.3299
#>  0.2730 -2.1429  2.1008
#> -0.4236 -1.4043  0.9649
#>  0.7149 -0.1239  0.1402
#> -0.0684 -0.8429  0.9603
#> -0.9608 -2.1126 -1.3324
#> -0.8603  0.5753 -0.8214
#> -1.7250  0.7079 -0.5922
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.7277  0.2215 -1.3676  0.8381
#> -1.5191 -2.4287 -1.3709 -0.2237
#>  0.5587  0.5782 -2.0714  0.0610
#>  1.4711  2.0462 -0.1107 -0.7669
#> -0.2942 -0.4251 -1.9183 -1.1335
#> -0.3573 -0.6660  0.5807  1.0539
#>  0.6083  0.4298 -0.1961 -0.2204
#> -0.4437  0.3918  0.3036 -1.3575
#> -0.7293 -0.0699  0.2836  0.2420
#> -0.6383  0.1342 -0.2752 -1.2052
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.6610
#>  0.1783
#>  2.9860
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.6098
#>  1.1380
#>  0.2154
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.1705
#>  0.8160
#>  0.3299
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.2730
#> -2.1429
#>  2.1008
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.4236
#> -1.4043
#>  0.9649
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.7149
#> -0.1239
#>  0.1402
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.0684
#> -0.8429
#>  0.9603
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.9608
#> -2.1126
#> -1.3324
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.8603
#>  0.5753
#> -0.8214
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.7250
#>  0.7079
#> -0.5922
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.7277
#>  0.2215
#> -1.3676
#>  0.8381
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.5191
#> -2.4287
#> -1.3709
#> -0.2237
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5587
#>  0.5782
#> -2.0714
#>  0.0610
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.4711
#>  2.0462
#> -0.1107
#> -0.7669
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.2942
#> -0.4251
#> -1.9183
#> -1.1335
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.3573
#> -0.6660
#>  0.5807
#>  1.0539
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.6083
#>  0.4298
#> -0.1961
#> -0.2204
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.4437
#>  0.3918
#>  0.3036
#> -1.3575
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.7293
#> -0.0699
#>  0.2836
#>  0.2420
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.6383
#>  0.1342
#> -0.2752
#> -1.2052
#> [ CPUFloatType{4} ]
#> 
#> 
```
