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
#>  1.0705 -0.4093  1.4684
#>  1.9242 -1.8017 -0.6537
#>  1.0277  1.1841  1.8040
#>  0.7723 -0.8116 -1.0479
#> -0.2386 -1.9887  0.0158
#> -0.6123 -1.3558 -1.4855
#>  0.2559  0.6241 -1.0829
#>  1.0271 -1.3633 -1.2667
#>  1.5683 -0.2236  1.1668
#>  0.5246 -0.6189 -0.7345
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.0705
#> -0.4093
#>  1.4684
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.9242
#> -1.8017
#> -0.6537
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.0277
#>  1.1841
#>  1.8040
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7723
#> -0.8116
#> -1.0479
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2386
#> -1.9887
#>  0.0158
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.6123
#> -1.3558
#> -1.4855
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.2559
#>  0.6241
#> -1.0829
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.0271
#> -1.3633
#> -1.2667
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.5683
#> -0.2236
#>  1.1668
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.5246
#> -0.6189
#> -0.7345
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.0705 -0.4093  1.4684
#>  1.9242 -1.8017 -0.6537
#>  1.0277  1.1841  1.8040
#>  0.7723 -0.8116 -1.0479
#> -0.2386 -1.9887  0.0158
#> -0.6123 -1.3558 -1.4855
#>  0.2559  0.6241 -1.0829
#>  1.0271 -1.3633 -1.2667
#>  1.5683 -0.2236  1.1668
#>  0.5246 -0.6189 -0.7345
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2337  0.2194  0.5066 -1.5210
#>  0.3437  0.2658 -1.6275 -0.6004
#>  0.1509 -0.2844 -1.2588  0.4453
#>  0.7229  1.1651  0.4448 -2.3959
#> -2.0451  0.0531  1.3607  0.2920
#>  0.4016  0.4357 -0.2849  0.0481
#> -0.2230  1.2774  0.4348  1.4830
#> -2.7458  0.8774  0.2094 -1.7778
#> -0.4005 -1.0095 -0.0727  0.1764
#>  2.0321  2.1209  1.0325  1.5886
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.0705
#> -0.4093
#>  1.4684
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.9242
#> -1.8017
#> -0.6537
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.0277
#>  1.1841
#>  1.8040
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7723
#> -0.8116
#> -1.0479
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2386
#> -1.9887
#>  0.0158
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.6123
#> -1.3558
#> -1.4855
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.2559
#>  0.6241
#> -1.0829
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.0271
#> -1.3633
#> -1.2667
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.5683
#> -0.2236
#>  1.1668
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.5246
#> -0.6189
#> -0.7345
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2337
#>  0.2194
#>  0.5066
#> -1.5210
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.3437
#>  0.2658
#> -1.6275
#> -0.6004
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.1509
#> -0.2844
#> -1.2588
#>  0.4453
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.7229
#>  1.1651
#>  0.4448
#> -2.3959
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -2.0451
#>  0.0531
#>  1.3607
#>  0.2920
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4016
#>  0.4357
#> -0.2849
#>  0.0481
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.2230
#>  1.2774
#>  0.4348
#>  1.4830
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -2.7458
#>  0.8774
#>  0.2094
#> -1.7778
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.4005
#> -1.0095
#> -0.0727
#>  0.1764
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  2.0321
#>  2.1209
#>  1.0325
#>  1.5886
#> [ CPUFloatType{4} ]
#> 
#> 
```
