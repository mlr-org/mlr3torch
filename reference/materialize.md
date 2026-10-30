# Materialize Lazy Tensor Columns

This will materialize a
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
or a [`data.frame()`](https://rdrr.io/r/base/data.frame.html) /
[`list()`](https://rdrr.io/r/base/list.html) containing – among other
things –
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
columns. I.e. the data described in the underlying
[`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)s
is loaded for the indices in the
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md),
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
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
  or a [`list()`](https://rdrr.io/r/base/list.html) /
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html) containing
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md).

2.  Processing these batches in the preprocessing
    [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s.

3.  Returning the result of the
    [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
    pointed to by the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)
    (`pointer`).

With multiple
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
#> -0.3914  0.5998 -0.6871
#>  0.1128 -0.3246  0.2985
#> -0.8081 -1.9191 -0.6294
#>  0.1903 -1.9602  1.1363
#>  1.1221  1.8635  0.1138
#>  0.6614 -0.2271  0.4806
#> -0.2046 -1.9403 -1.0873
#> -1.2077  0.9946 -0.6130
#> -1.5997 -2.4105  0.6211
#>  0.9457 -0.3065 -0.6982
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.3914
#>  0.5998
#> -0.6871
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.1128
#> -0.3246
#>  0.2985
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.8081
#> -1.9191
#> -0.6294
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.1903
#> -1.9602
#>  1.1363
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.1221
#>  1.8635
#>  0.1138
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.6614
#> -0.2271
#>  0.4806
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.2046
#> -1.9403
#> -1.0873
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.2077
#>  0.9946
#> -0.6130
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.5997
#> -2.4105
#>  0.6211
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.9457
#> -0.3065
#> -0.6982
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.3914  0.5998 -0.6871
#>  0.1128 -0.3246  0.2985
#> -0.8081 -1.9191 -0.6294
#>  0.1903 -1.9602  1.1363
#>  1.1221  1.8635  0.1138
#>  0.6614 -0.2271  0.4806
#> -0.2046 -1.9403 -1.0873
#> -1.2077  0.9946 -0.6130
#> -1.5997 -2.4105  0.6211
#>  0.9457 -0.3065 -0.6982
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.8751  0.5879  0.1231  2.5680
#>  0.1166  0.3282 -0.9205  0.4088
#>  0.8519 -1.1165 -0.8212 -1.4470
#>  0.3722  1.3718  0.2014  0.0493
#>  1.5255 -1.1163  0.2070  0.0407
#>  1.4153 -0.6770 -0.0618  2.5546
#> -0.8522 -0.5705 -0.0760 -1.3711
#> -0.6159  0.1945 -1.5117 -0.0796
#> -0.2897 -0.9398 -0.8700  0.2539
#> -0.0406  1.3122  1.1406 -1.2558
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.3914
#>  0.5998
#> -0.6871
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.1128
#> -0.3246
#>  0.2985
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.8081
#> -1.9191
#> -0.6294
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.1903
#> -1.9602
#>  1.1363
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.1221
#>  1.8635
#>  0.1138
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.6614
#> -0.2271
#>  0.4806
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.2046
#> -1.9403
#> -1.0873
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.2077
#>  0.9946
#> -0.6130
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.5997
#> -2.4105
#>  0.6211
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.9457
#> -0.3065
#> -0.6982
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.8751
#>  0.5879
#>  0.1231
#>  2.5680
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.1166
#>  0.3282
#> -0.9205
#>  0.4088
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.8519
#> -1.1165
#> -0.8212
#> -1.4470
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.3722
#>  1.3718
#>  0.2014
#>  0.0493
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.5255
#> -1.1163
#>  0.2070
#>  0.0407
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.4153
#> -0.6770
#> -0.0618
#>  2.5546
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.8522
#> -0.5705
#> -0.0760
#> -1.3711
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.6159
#>  0.1945
#> -1.5117
#> -0.0796
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.2897
#> -0.9398
#> -0.8700
#>  0.2539
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.0406
#>  1.3122
#>  1.1406
#> -1.2558
#> [ CPUFloatType{4} ]
#> 
#> 
```
