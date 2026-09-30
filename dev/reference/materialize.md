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
#>  0.6912 -1.6538  0.0820
#> -1.1560 -0.3822  1.8380
#>  1.0133 -0.1384 -0.0946
#> -1.2683  0.5465 -0.9152
#> -2.2033  0.6751  1.5544
#>  1.2371  0.2528  0.3527
#>  1.3974 -0.8263  0.2368
#>  0.3885 -1.5706  0.0921
#>  0.1286 -0.8045 -0.8368
#> -0.5536 -0.7878 -0.0297
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.6912
#> -1.6538
#>  0.0820
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.1560
#> -0.3822
#>  1.8380
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.0133
#> -0.1384
#> -0.0946
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.2683
#>  0.5465
#> -0.9152
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -2.2033
#>  0.6751
#>  1.5544
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.2371
#>  0.2528
#>  0.3527
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.3974
#> -0.8263
#>  0.2368
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.3885
#> -1.5706
#>  0.0921
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.1286
#> -0.8045
#> -0.8368
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.5536
#> -0.7878
#> -0.0297
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.6912 -1.6538  0.0820
#> -1.1560 -0.3822  1.8380
#>  1.0133 -0.1384 -0.0946
#> -1.2683  0.5465 -0.9152
#> -2.2033  0.6751  1.5544
#>  1.2371  0.2528  0.3527
#>  1.3974 -0.8263  0.2368
#>  0.3885 -1.5706  0.0921
#>  0.1286 -0.8045 -0.8368
#> -0.5536 -0.7878 -0.0297
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.6061 -1.3329 -0.2059  2.1082
#> -1.1554 -1.2768 -2.0917  0.6783
#> -0.3991  0.4288  1.5058  0.7573
#>  1.4459  0.8809  0.6362  0.5946
#>  0.6125  2.2450 -1.3714 -0.1669
#> -0.1878  1.6882  0.5290  0.6857
#> -1.1030  0.2293 -0.8703  1.6334
#> -0.5166  1.6014  0.0313 -0.2366
#>  1.8638 -0.5666  0.3558  0.1972
#> -0.5351  0.2769  0.4136 -1.3227
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.6912
#> -1.6538
#>  0.0820
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.1560
#> -0.3822
#>  1.8380
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.0133
#> -0.1384
#> -0.0946
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.2683
#>  0.5465
#> -0.9152
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -2.2033
#>  0.6751
#>  1.5544
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.2371
#>  0.2528
#>  0.3527
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.3974
#> -0.8263
#>  0.2368
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.3885
#> -1.5706
#>  0.0921
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.1286
#> -0.8045
#> -0.8368
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.5536
#> -0.7878
#> -0.0297
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.6061
#> -1.3329
#> -0.2059
#>  2.1082
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.1554
#> -1.2768
#> -2.0917
#>  0.6783
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.3991
#>  0.4288
#>  1.5058
#>  0.7573
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.4459
#>  0.8809
#>  0.6362
#>  0.5946
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.6125
#>  2.2450
#> -1.3714
#> -0.1669
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1878
#>  1.6882
#>  0.5290
#>  0.6857
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.1030
#>  0.2293
#> -0.8703
#>  1.6334
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.5166
#>  1.6014
#>  0.0313
#> -0.2366
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.8638
#> -0.5666
#>  0.3558
#>  0.1972
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.5351
#>  0.2769
#>  0.4136
#> -1.3227
#> [ CPUFloatType{4} ]
#> 
#> 
```
