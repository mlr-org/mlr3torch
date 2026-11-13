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
#> -0.1044 -0.9621  0.8817
#> -1.1527 -1.9289 -0.4710
#> -0.2620 -0.5864 -0.9821
#>  0.2505  1.4575 -0.8036
#> -2.3156 -0.1182 -1.4066
#> -0.4524  0.5266 -0.5906
#> -0.9245  0.6654  1.0120
#>  0.7023 -0.6652 -0.3766
#>  1.8419  1.7512 -0.5226
#>  0.1341  0.2264 -0.6199
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.1044
#> -0.9621
#>  0.8817
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.1527
#> -1.9289
#> -0.4710
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.2620
#> -0.5864
#> -0.9821
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.2505
#>  1.4575
#> -0.8036
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -2.3156
#> -0.1182
#> -1.4066
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.4524
#>  0.5266
#> -0.5906
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.9245
#>  0.6654
#>  1.0120
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.7023
#> -0.6652
#> -0.3766
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.8419
#>  1.7512
#> -0.5226
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.1341
#>  0.2264
#> -0.6199
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.1044 -0.9621  0.8817
#> -1.1527 -1.9289 -0.4710
#> -0.2620 -0.5864 -0.9821
#>  0.2505  1.4575 -0.8036
#> -2.3156 -0.1182 -1.4066
#> -0.4524  0.5266 -0.5906
#> -0.9245  0.6654  1.0120
#>  0.7023 -0.6652 -0.3766
#>  1.8419  1.7512 -0.5226
#>  0.1341  0.2264 -0.6199
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.8397 -0.3993  0.9192  1.4694
#>  0.2828 -1.6387  0.7127 -0.0766
#> -0.0344 -0.8013 -0.6915  0.5012
#> -0.2615 -2.1255  0.8703 -1.2612
#>  1.4813  0.5875 -0.6289  1.0885
#>  0.5704  0.7046  0.1294 -1.0920
#> -0.9740  0.4240  0.7423  1.3162
#>  0.6077  0.8104  0.9521  0.0619
#> -0.3233 -0.4387  0.6396 -1.0164
#>  0.7335  0.5136 -0.1955  0.4423
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.1044
#> -0.9621
#>  0.8817
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.1527
#> -1.9289
#> -0.4710
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.2620
#> -0.5864
#> -0.9821
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.2505
#>  1.4575
#> -0.8036
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -2.3156
#> -0.1182
#> -1.4066
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.4524
#>  0.5266
#> -0.5906
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.9245
#>  0.6654
#>  1.0120
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.7023
#> -0.6652
#> -0.3766
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.8419
#>  1.7512
#> -0.5226
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.1341
#>  0.2264
#> -0.6199
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.8397
#> -0.3993
#>  0.9192
#>  1.4694
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.2828
#> -1.6387
#>  0.7127
#> -0.0766
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.0344
#> -0.8013
#> -0.6915
#>  0.5012
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.2615
#> -2.1255
#>  0.8703
#> -1.2612
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.4813
#>  0.5875
#> -0.6289
#>  1.0885
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.5704
#>  0.7046
#>  0.1294
#> -1.0920
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.9740
#>  0.4240
#>  0.7423
#>  1.3162
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.6077
#>  0.8104
#>  0.9521
#>  0.0619
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.3233
#> -0.4387
#>  0.6396
#> -1.0164
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.7335
#>  0.5136
#> -0.1955
#>  0.4423
#> [ CPUFloatType{4} ]
#> 
#> 
```
