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
#> -0.2183  0.0765 -0.3292
#>  0.6228 -0.3547 -0.4058
#> -0.0016 -0.7074  0.4377
#>  0.7633  0.2892  0.8720
#>  1.0408  0.4545 -0.4077
#>  0.6900 -1.1411 -0.4853
#> -0.9170  0.6842  1.3491
#> -0.9724  0.4946  0.5250
#> -2.2283 -0.3183 -2.4209
#>  1.2066  1.3069  0.9933
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.2183
#>  0.0765
#> -0.3292
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.6228
#> -0.3547
#> -0.4058
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.0016
#> -0.7074
#>  0.4377
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7633
#>  0.2892
#>  0.8720
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.0408
#>  0.4545
#> -0.4077
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.6900
#> -1.1411
#> -0.4853
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.9170
#>  0.6842
#>  1.3491
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.9724
#>  0.4946
#>  0.5250
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -2.2283
#> -0.3183
#> -2.4209
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.2066
#>  1.3069
#>  0.9933
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.2183  0.0765 -0.3292
#>  0.6228 -0.3547 -0.4058
#> -0.0016 -0.7074  0.4377
#>  0.7633  0.2892  0.8720
#>  1.0408  0.4545 -0.4077
#>  0.6900 -1.1411 -0.4853
#> -0.9170  0.6842  1.3491
#> -0.9724  0.4946  0.5250
#> -2.2283 -0.3183 -2.4209
#>  1.2066  1.3069  0.9933
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.9702 -0.4197  1.3060 -0.3120
#>  0.0903 -0.0877  1.9765 -0.1625
#> -0.6706 -0.1488  1.1724  0.1588
#>  1.4586 -0.4618  1.3617  0.2457
#>  1.5170 -0.1186 -1.4404 -0.1776
#> -1.5691 -0.2523  1.0782  1.4924
#> -0.5485  0.3051  0.1735  0.8472
#>  0.4105  0.3741 -2.1173 -0.4728
#>  0.6670  1.0781  0.4515 -0.0717
#> -0.1657 -1.0830 -1.7542 -0.5070
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.2183
#>  0.0765
#> -0.3292
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.6228
#> -0.3547
#> -0.4058
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.0016
#> -0.7074
#>  0.4377
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7633
#>  0.2892
#>  0.8720
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.0408
#>  0.4545
#> -0.4077
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.6900
#> -1.1411
#> -0.4853
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.9170
#>  0.6842
#>  1.3491
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.9724
#>  0.4946
#>  0.5250
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -2.2283
#> -0.3183
#> -2.4209
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.2066
#>  1.3069
#>  0.9933
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.9702
#> -0.4197
#>  1.3060
#> -0.3120
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.0903
#> -0.0877
#>  1.9765
#> -0.1625
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.6706
#> -0.1488
#>  1.1724
#>  0.1588
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.4586
#> -0.4618
#>  1.3617
#>  0.2457
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.5170
#> -0.1186
#> -1.4404
#> -0.1776
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.5691
#> -0.2523
#>  1.0782
#>  1.4924
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.5485
#>  0.3051
#>  0.1735
#>  0.8472
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4105
#>  0.3741
#> -2.1173
#> -0.4728
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.6670
#>  1.0781
#>  0.4515
#> -0.0717
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.1657
#> -1.0830
#> -1.7542
#> -0.5070
#> [ CPUFloatType{4} ]
#> 
#> 
```
