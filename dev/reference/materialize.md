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
#> -0.2720  1.5360  1.2835
#> -0.6877 -1.5627 -0.1691
#>  0.4105  1.1808 -2.1380
#> -0.8174 -0.5111 -0.9181
#> -0.8890  0.2818  0.5072
#>  2.8750 -1.7031 -1.2660
#> -0.7052  0.5042 -1.4185
#> -1.7208  0.9066  1.0193
#> -1.0001  0.4224 -0.9009
#> -0.0856  0.5273  0.6377
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.2720
#>  1.5360
#>  1.2835
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.6877
#> -1.5627
#> -0.1691
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.4105
#>  1.1808
#> -2.1380
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.8174
#> -0.5111
#> -0.9181
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.8890
#>  0.2818
#>  0.5072
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  2.8750
#> -1.7031
#> -1.2660
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.7052
#>  0.5042
#> -1.4185
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.7208
#>  0.9066
#>  1.0193
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.0001
#>  0.4224
#> -0.9009
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.0856
#>  0.5273
#>  0.6377
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.2720  1.5360  1.2835
#> -0.6877 -1.5627 -0.1691
#>  0.4105  1.1808 -2.1380
#> -0.8174 -0.5111 -0.9181
#> -0.8890  0.2818  0.5072
#>  2.8750 -1.7031 -1.2660
#> -0.7052  0.5042 -1.4185
#> -1.7208  0.9066  1.0193
#> -1.0001  0.4224 -0.9009
#> -0.0856  0.5273  0.6377
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.3853  0.7656 -1.6646  1.5725
#>  0.1517 -0.5377  1.4898  0.4881
#> -0.1904  0.0551  0.7521 -2.1370
#>  0.0959 -0.3127 -1.0329  0.8978
#>  0.0644 -0.3983  2.3657 -0.8545
#>  0.4798 -0.1880 -0.7911 -1.0787
#> -0.9285 -1.7432  0.9714 -0.3258
#>  1.1875  0.7382 -1.7923  1.4269
#>  1.1527 -1.2604 -1.3414 -1.7401
#> -0.1304  0.1351  1.3694 -0.3764
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.2720
#>  1.5360
#>  1.2835
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.6877
#> -1.5627
#> -0.1691
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.4105
#>  1.1808
#> -2.1380
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.8174
#> -0.5111
#> -0.9181
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.8890
#>  0.2818
#>  0.5072
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  2.8750
#> -1.7031
#> -1.2660
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.7052
#>  0.5042
#> -1.4185
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.7208
#>  0.9066
#>  1.0193
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.0001
#>  0.4224
#> -0.9009
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.0856
#>  0.5273
#>  0.6377
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.3853
#>  0.7656
#> -1.6646
#>  1.5725
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.1517
#> -0.5377
#>  1.4898
#>  0.4881
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.1904
#>  0.0551
#>  0.7521
#> -2.1370
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.0959
#> -0.3127
#> -1.0329
#>  0.8978
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.0644
#> -0.3983
#>  2.3657
#> -0.8545
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4798
#> -0.1880
#> -0.7911
#> -1.0787
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.9285
#> -1.7432
#>  0.9714
#> -0.3258
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.1875
#>  0.7382
#> -1.7923
#>  1.4269
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.1527
#> -1.2604
#> -1.3414
#> -1.7401
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.1304
#>  0.1351
#>  1.3694
#> -0.3764
#> [ CPUFloatType{4} ]
#> 
#> 
```
