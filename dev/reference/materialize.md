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
#> -0.6343  0.1521 -0.4365
#> -0.3404 -1.8116 -0.2409
#>  0.6326  1.1930 -0.1058
#>  0.0826 -1.1342  1.1685
#>  0.3257 -1.4807 -0.0066
#>  0.9760  0.4995  0.0409
#> -0.1234 -0.2086 -1.1064
#>  0.8152 -0.7993  0.5119
#>  1.2337  0.1067 -0.1970
#>  0.5940  0.5931 -0.1958
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.6343
#>  0.1521
#> -0.4365
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3404
#> -1.8116
#> -0.2409
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.6326
#>  1.1930
#> -0.1058
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.0826
#> -1.1342
#>  1.1685
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.3257
#> -1.4807
#> -0.0066
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.9760
#>  0.4995
#>  0.0409
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.1234
#> -0.2086
#> -1.1064
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.8152
#> -0.7993
#>  0.5119
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.2337
#>  0.1067
#> -0.1970
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.5940
#>  0.5931
#> -0.1958
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.6343  0.1521 -0.4365
#> -0.3404 -1.8116 -0.2409
#>  0.6326  1.1930 -0.1058
#>  0.0826 -1.1342  1.1685
#>  0.3257 -1.4807 -0.0066
#>  0.9760  0.4995  0.0409
#> -0.1234 -0.2086 -1.1064
#>  0.8152 -0.7993  0.5119
#>  1.2337  0.1067 -0.1970
#>  0.5940  0.5931 -0.1958
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.8124 -1.3641  1.2103 -1.0939
#> -0.1697  0.1276 -0.8540  2.2473
#>  0.1642 -0.1709 -0.2377  0.6046
#>  0.9128  0.9306  0.0718  2.5140
#>  3.0806 -1.2764 -1.2065 -0.3481
#> -0.1702  0.0507  1.0202  0.8131
#>  0.6955  0.7489  0.0480 -0.6081
#> -0.1847  0.1505 -0.1827  1.4014
#>  0.6397  0.2358 -0.7452  0.8744
#> -0.4289  0.5241 -0.0857  1.6771
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.6343
#>  0.1521
#> -0.4365
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3404
#> -1.8116
#> -0.2409
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.6326
#>  1.1930
#> -0.1058
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.0826
#> -1.1342
#>  1.1685
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.3257
#> -1.4807
#> -0.0066
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.9760
#>  0.4995
#>  0.0409
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.1234
#> -0.2086
#> -1.1064
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.8152
#> -0.7993
#>  0.5119
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.2337
#>  0.1067
#> -0.1970
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.5940
#>  0.5931
#> -0.1958
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.8124
#> -1.3641
#>  1.2103
#> -1.0939
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.1697
#>  0.1276
#> -0.8540
#>  2.2473
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.1642
#> -0.1709
#> -0.2377
#>  0.6046
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.9128
#>  0.9306
#>  0.0718
#>  2.5140
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  3.0806
#> -1.2764
#> -1.2065
#> -0.3481
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1702
#>  0.0507
#>  1.0202
#>  0.8131
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.6955
#>  0.7489
#>  0.0480
#> -0.6081
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.1847
#>  0.1505
#> -0.1827
#>  1.4014
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.6397
#>  0.2358
#> -0.7452
#>  0.8744
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.4289
#>  0.5241
#> -0.0857
#>  1.6771
#> [ CPUFloatType{4} ]
#> 
#> 
```
