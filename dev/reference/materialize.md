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
#> -0.3045  1.9089 -0.9203
#> -1.1570 -0.5081  0.3259
#>  0.2910 -0.1892 -0.5499
#> -0.4652 -0.9140  0.1581
#>  0.1445  1.5441 -0.7432
#>  1.0297 -0.1498  1.4768
#> -2.3158 -0.4582  0.7755
#>  1.3081 -1.0340 -0.6434
#>  0.6369  0.6766 -0.9762
#> -0.8103 -0.1636  0.2964
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.3045
#>  1.9089
#> -0.9203
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.1570
#> -0.5081
#>  0.3259
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.2910
#> -0.1892
#> -0.5499
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.4652
#> -0.9140
#>  0.1581
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.1445
#>  1.5441
#> -0.7432
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.0297
#> -0.1498
#>  1.4768
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -2.3158
#> -0.4582
#>  0.7755
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.3081
#> -1.0340
#> -0.6434
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.6369
#>  0.6766
#> -0.9762
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.8103
#> -0.1636
#>  0.2964
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.3045  1.9089 -0.9203
#> -1.1570 -0.5081  0.3259
#>  0.2910 -0.1892 -0.5499
#> -0.4652 -0.9140  0.1581
#>  0.1445  1.5441 -0.7432
#>  1.0297 -0.1498  1.4768
#> -2.3158 -0.4582  0.7755
#>  1.3081 -1.0340 -0.6434
#>  0.6369  0.6766 -0.9762
#> -0.8103 -0.1636  0.2964
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.9391 -0.6224  0.1010 -1.0996
#>  0.1238  0.4465  0.6721  0.1628
#> -0.1391 -0.7988 -0.3711 -0.2141
#> -0.0597 -0.8090 -1.6347  0.2396
#> -0.3242  0.0211 -0.7826  0.5092
#> -0.1995 -0.6326  1.4579 -0.9824
#>  1.0308 -1.2487 -0.5219 -1.4633
#>  0.7323  0.0044 -0.5164 -0.8499
#> -1.1129 -0.4191  0.4094  1.5517
#>  0.7882  0.9256  0.9293 -0.5911
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.3045
#>  1.9089
#> -0.9203
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.1570
#> -0.5081
#>  0.3259
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.2910
#> -0.1892
#> -0.5499
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.4652
#> -0.9140
#>  0.1581
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.1445
#>  1.5441
#> -0.7432
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.0297
#> -0.1498
#>  1.4768
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -2.3158
#> -0.4582
#>  0.7755
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.3081
#> -1.0340
#> -0.6434
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.6369
#>  0.6766
#> -0.9762
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.8103
#> -0.1636
#>  0.2964
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.9391
#> -0.6224
#>  0.1010
#> -1.0996
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.1238
#>  0.4465
#>  0.6721
#>  0.1628
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.1391
#> -0.7988
#> -0.3711
#> -0.2141
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.0597
#> -0.8090
#> -1.6347
#>  0.2396
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.3242
#>  0.0211
#> -0.7826
#>  0.5092
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1995
#> -0.6326
#>  1.4579
#> -0.9824
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  1.0308
#> -1.2487
#> -0.5219
#> -1.4633
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.7323
#>  0.0044
#> -0.5164
#> -0.8499
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.1129
#> -0.4191
#>  0.4094
#>  1.5517
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.7882
#>  0.9256
#>  0.9293
#> -0.5911
#> [ CPUFloatType{4} ]
#> 
#> 
```
