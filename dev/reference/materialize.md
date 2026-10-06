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
#> -0.4314  0.1759  0.3345
#>  0.6513  2.8719  0.9378
#> -0.0601  1.5801  0.1079
#> -0.0931  1.7542 -0.3105
#>  0.7758  0.5859  0.9457
#>  0.2296 -0.5534  0.1716
#> -0.0256  0.8912 -0.0910
#>  1.6812  1.4119  1.6907
#> -0.3854  1.1121  1.2129
#> -0.8378  1.0568  0.6089
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.4314
#>  0.1759
#>  0.3345
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.6513
#>  2.8719
#>  0.9378
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.0601
#>  1.5801
#>  0.1079
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.0931
#>  1.7542
#> -0.3105
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.7758
#>  0.5859
#>  0.9457
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.2296
#> -0.5534
#>  0.1716
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.0256
#>  0.8912
#> -0.0910
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.6812
#>  1.4119
#>  1.6907
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.3854
#>  1.1121
#>  1.2129
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.8378
#>  1.0568
#>  0.6089
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.4314  0.1759  0.3345
#>  0.6513  2.8719  0.9378
#> -0.0601  1.5801  0.1079
#> -0.0931  1.7542 -0.3105
#>  0.7758  0.5859  0.9457
#>  0.2296 -0.5534  0.1716
#> -0.0256  0.8912 -0.0910
#>  1.6812  1.4119  1.6907
#> -0.3854  1.1121  1.2129
#> -0.8378  1.0568  0.6089
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.1933  0.8778 -0.1790  0.5598
#>  0.8763  0.9388  1.2518 -0.7617
#> -0.8305  0.2762  0.5963 -1.1195
#> -1.3951  2.0442 -0.0705  1.8034
#> -0.1904 -1.8009  1.2310 -1.5247
#> -0.6751  1.0847 -1.7899  0.5341
#>  1.0398 -1.5128 -0.3152 -0.6642
#> -1.1293 -0.5396 -1.2121  0.3345
#> -0.2904 -2.3197 -0.4391  0.4367
#>  0.4499 -0.2300  1.0267  0.1838
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.4314
#>  0.1759
#>  0.3345
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.6513
#>  2.8719
#>  0.9378
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.0601
#>  1.5801
#>  0.1079
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.0931
#>  1.7542
#> -0.3105
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.7758
#>  0.5859
#>  0.9457
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.2296
#> -0.5534
#>  0.1716
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.0256
#>  0.8912
#> -0.0910
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.6812
#>  1.4119
#>  1.6907
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.3854
#>  1.1121
#>  1.2129
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.8378
#>  1.0568
#>  0.6089
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.1933
#>  0.8778
#> -0.1790
#>  0.5598
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.8763
#>  0.9388
#>  1.2518
#> -0.7617
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.8305
#>  0.2762
#>  0.5963
#> -1.1195
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.3951
#>  2.0442
#> -0.0705
#>  1.8034
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.1904
#> -1.8009
#>  1.2310
#> -1.5247
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.6751
#>  1.0847
#> -1.7899
#>  0.5341
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  1.0398
#> -1.5128
#> -0.3152
#> -0.6642
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.1293
#> -0.5396
#> -1.2121
#>  0.3345
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.2904
#> -2.3197
#> -0.4391
#>  0.4367
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.4499
#> -0.2300
#>  1.0267
#>  0.1838
#> [ CPUFloatType{4} ]
#> 
#> 
```
