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
#> -1.6446 -0.4035  1.1081
#>  2.4934  0.6574  0.0190
#> -0.2721  0.2813  0.0253
#>  0.4300 -0.1167 -0.1694
#>  0.9125 -0.3401  1.3831
#>  0.4926  0.9101 -0.3480
#> -1.7086  1.2894 -0.7238
#> -0.9563 -0.8526 -0.0725
#> -0.6734  1.1568 -0.4206
#> -1.5177 -1.7435 -0.4782
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.6446
#> -0.4035
#>  1.1081
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  2.4934
#>  0.6574
#>  0.0190
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.2721
#>  0.2813
#>  0.0253
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.4300
#> -0.1167
#> -0.1694
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.9125
#> -0.3401
#>  1.3831
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.4926
#>  0.9101
#> -0.3480
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -1.7086
#>  1.2894
#> -0.7238
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.9563
#> -0.8526
#> -0.0725
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.6734
#>  1.1568
#> -0.4206
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.5177
#> -1.7435
#> -0.4782
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.6446 -0.4035  1.1081
#>  2.4934  0.6574  0.0190
#> -0.2721  0.2813  0.0253
#>  0.4300 -0.1167 -0.1694
#>  0.9125 -0.3401  1.3831
#>  0.4926  0.9101 -0.3480
#> -1.7086  1.2894 -0.7238
#> -0.9563 -0.8526 -0.0725
#> -0.6734  1.1568 -0.4206
#> -1.5177 -1.7435 -0.4782
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.8004  0.3448 -1.7166  0.0550
#>  0.1774  1.5837  0.7517 -0.5701
#> -1.0847 -1.4359  1.3756  2.2663
#>  0.6918  0.6269 -1.5200 -0.5964
#> -1.8260 -1.2084 -0.9444 -0.3168
#> -1.6272 -0.1248  0.4189  1.2784
#>  0.8020 -1.1451 -1.5013 -1.0152
#> -0.5301  0.6917 -1.0879 -0.1354
#> -0.8348  1.1200  0.6872  0.3990
#>  0.9558 -0.1658 -0.4564 -0.8936
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.6446
#> -0.4035
#>  1.1081
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  2.4934
#>  0.6574
#>  0.0190
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.2721
#>  0.2813
#>  0.0253
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.4300
#> -0.1167
#> -0.1694
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.9125
#> -0.3401
#>  1.3831
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.4926
#>  0.9101
#> -0.3480
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -1.7086
#>  1.2894
#> -0.7238
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.9563
#> -0.8526
#> -0.0725
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.6734
#>  1.1568
#> -0.4206
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.5177
#> -1.7435
#> -0.4782
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.8004
#>  0.3448
#> -1.7166
#>  0.0550
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.1774
#>  1.5837
#>  0.7517
#> -0.5701
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.0847
#> -1.4359
#>  1.3756
#>  2.2663
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.6918
#>  0.6269
#> -1.5200
#> -0.5964
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.8260
#> -1.2084
#> -0.9444
#> -0.3168
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.6272
#> -0.1248
#>  0.4189
#>  1.2784
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.8020
#> -1.1451
#> -1.5013
#> -1.0152
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.5301
#>  0.6917
#> -1.0879
#> -0.1354
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.8348
#>  1.1200
#>  0.6872
#>  0.3990
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.9558
#> -0.1658
#> -0.4564
#> -0.8936
#> [ CPUFloatType{4} ]
#> 
#> 
```
