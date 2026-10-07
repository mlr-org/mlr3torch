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
#> -1.1658 -1.5582 -0.4538
#> -0.3966  0.7746 -0.2495
#> -0.1310 -0.5282 -0.8799
#>  0.2581  1.0375 -2.1893
#> -0.8415 -1.7171  0.1064
#>  0.5272  0.1353  0.0229
#> -0.4247 -0.8309  0.1166
#>  0.9986 -0.0411  0.1972
#> -1.6733  1.0713  1.2373
#>  0.3175  1.2539 -0.8581
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.1658
#> -1.5582
#> -0.4538
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3966
#>  0.7746
#> -0.2495
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.1310
#> -0.5282
#> -0.8799
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.2581
#>  1.0375
#> -2.1893
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.8415
#> -1.7171
#>  0.1064
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.5272
#>  0.1353
#>  0.0229
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.4247
#> -0.8309
#>  0.1166
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.9986
#> -0.0411
#>  0.1972
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.6733
#>  1.0713
#>  1.2373
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.3175
#>  1.2539
#> -0.8581
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.1658 -1.5582 -0.4538
#> -0.3966  0.7746 -0.2495
#> -0.1310 -0.5282 -0.8799
#>  0.2581  1.0375 -2.1893
#> -0.8415 -1.7171  0.1064
#>  0.5272  0.1353  0.0229
#> -0.4247 -0.8309  0.1166
#>  0.9986 -0.0411  0.1972
#> -1.6733  1.0713  1.2373
#>  0.3175  1.2539 -0.8581
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.1209  0.1889 -0.6462 -0.5707
#>  0.9506 -0.9418  0.9367  0.3607
#> -1.5779 -0.0366 -1.5645 -1.1454
#> -1.3645  0.0793  0.0175  0.1774
#>  0.4787  0.3701 -1.6211  1.4149
#> -1.4595  2.3834  0.4597 -0.5103
#> -0.7347 -1.0197  1.3639  0.4335
#>  0.4586  2.3136  0.0647 -0.3819
#>  0.6508  1.1772 -1.3990 -1.5010
#>  0.1241  0.0240 -1.8653  0.5542
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.1658
#> -1.5582
#> -0.4538
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3966
#>  0.7746
#> -0.2495
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.1310
#> -0.5282
#> -0.8799
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.2581
#>  1.0375
#> -2.1893
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.8415
#> -1.7171
#>  0.1064
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.5272
#>  0.1353
#>  0.0229
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.4247
#> -0.8309
#>  0.1166
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.9986
#> -0.0411
#>  0.1972
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.6733
#>  1.0713
#>  1.2373
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.3175
#>  1.2539
#> -0.8581
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.1209
#>  0.1889
#> -0.6462
#> -0.5707
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.9506
#> -0.9418
#>  0.9367
#>  0.3607
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.5779
#> -0.0366
#> -1.5645
#> -1.1454
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.3645
#>  0.0793
#>  0.0175
#>  0.1774
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.4787
#>  0.3701
#> -1.6211
#>  1.4149
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.4595
#>  2.3834
#>  0.4597
#> -0.5103
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.7347
#> -1.0197
#>  1.3639
#>  0.4335
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4586
#>  2.3136
#>  0.0647
#> -0.3819
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.6508
#>  1.1772
#> -1.3990
#> -1.5010
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.1241
#>  0.0240
#> -1.8653
#>  0.5542
#> [ CPUFloatType{4} ]
#> 
#> 
```
