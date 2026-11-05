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
#> -0.1804 -1.5225  0.4179
#> -1.3954 -1.1700 -0.0076
#> -1.9164 -1.4841  0.2548
#> -0.8177 -0.0989  0.3626
#> -0.2235 -0.5691  0.5642
#>  1.4967  0.6006 -0.9133
#> -1.0582  3.2437  0.9337
#>  0.6781  1.2949 -1.0532
#>  1.2292  0.6579  1.9831
#> -0.2178  0.7281 -0.5520
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.1804
#> -1.5225
#>  0.4179
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.3954
#> -1.1700
#> -0.0076
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.9164
#> -1.4841
#>  0.2548
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.8177
#> -0.0989
#>  0.3626
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2235
#> -0.5691
#>  0.5642
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.4967
#>  0.6006
#> -0.9133
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -1.0582
#>  3.2437
#>  0.9337
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.6781
#>  1.2949
#> -1.0532
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.2292
#>  0.6579
#>  1.9831
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.2178
#>  0.7281
#> -0.5520
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.1804 -1.5225  0.4179
#> -1.3954 -1.1700 -0.0076
#> -1.9164 -1.4841  0.2548
#> -0.8177 -0.0989  0.3626
#> -0.2235 -0.5691  0.5642
#>  1.4967  0.6006 -0.9133
#> -1.0582  3.2437  0.9337
#>  0.6781  1.2949 -1.0532
#>  1.2292  0.6579  1.9831
#> -0.2178  0.7281 -0.5520
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.5947  1.9390 -0.3790  0.1577
#> -0.7514  0.9741  0.1917 -1.2042
#>  0.1378 -0.0679 -2.1522  0.1297
#> -2.3137 -0.7441 -0.3996  1.6312
#>  1.0380  0.0995 -0.3248 -1.8434
#>  0.3969  2.1976  0.3364 -0.1743
#> -0.4741 -1.0662  0.8469  0.6292
#> -0.5089 -0.5177 -0.2643  0.0733
#> -0.9589 -0.0324 -1.1256 -0.7233
#> -0.1070  1.9593 -1.3264  0.2269
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.1804
#> -1.5225
#>  0.4179
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.3954
#> -1.1700
#> -0.0076
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.9164
#> -1.4841
#>  0.2548
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.8177
#> -0.0989
#>  0.3626
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2235
#> -0.5691
#>  0.5642
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.4967
#>  0.6006
#> -0.9133
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -1.0582
#>  3.2437
#>  0.9337
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.6781
#>  1.2949
#> -1.0532
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.2292
#>  0.6579
#>  1.9831
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.2178
#>  0.7281
#> -0.5520
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.5947
#>  1.9390
#> -0.3790
#>  0.1577
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.7514
#>  0.9741
#>  0.1917
#> -1.2042
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.1378
#> -0.0679
#> -2.1522
#>  0.1297
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -2.3137
#> -0.7441
#> -0.3996
#>  1.6312
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.0380
#>  0.0995
#> -0.3248
#> -1.8434
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.3969
#>  2.1976
#>  0.3364
#> -0.1743
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.4741
#> -1.0662
#>  0.8469
#>  0.6292
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.5089
#> -0.5177
#> -0.2643
#>  0.0733
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.9589
#> -0.0324
#> -1.1256
#> -0.7233
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.1070
#>  1.9593
#> -1.3264
#>  0.2269
#> [ CPUFloatType{4} ]
#> 
#> 
```
