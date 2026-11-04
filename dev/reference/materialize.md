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
#> -0.8844 -0.0964 -0.6067
#>  0.5198  1.2186  1.4260
#>  1.1124  0.6556  2.6156
#> -0.3865  0.2369  0.8536
#> -1.2660  0.7218 -0.9317
#>  0.3408  1.3695 -0.6714
#> -0.8833  1.6252  0.3801
#> -0.5000 -0.3880  1.3117
#>  0.5339  0.8954 -1.7333
#>  1.0540  0.1057  1.5848
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.8844
#> -0.0964
#> -0.6067
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.5198
#>  1.2186
#>  1.4260
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.1124
#>  0.6556
#>  2.6156
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.3865
#>  0.2369
#>  0.8536
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.2660
#>  0.7218
#> -0.9317
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.3408
#>  1.3695
#> -0.6714
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.8833
#>  1.6252
#>  0.3801
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.5000
#> -0.3880
#>  1.3117
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.5339
#>  0.8954
#> -1.7333
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.0540
#>  0.1057
#>  1.5848
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.8844 -0.0964 -0.6067
#>  0.5198  1.2186  1.4260
#>  1.1124  0.6556  2.6156
#> -0.3865  0.2369  0.8536
#> -1.2660  0.7218 -0.9317
#>  0.3408  1.3695 -0.6714
#> -0.8833  1.6252  0.3801
#> -0.5000 -0.3880  1.3117
#>  0.5339  0.8954 -1.7333
#>  1.0540  0.1057  1.5848
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.2711 -0.8584  1.1657  0.0525
#> -0.9575 -0.8781 -1.8560  0.1529
#> -0.0188  0.1271  0.7161  0.8354
#>  1.4195  0.0223  2.7435  0.0151
#>  1.0067 -2.3031 -1.9115 -1.4543
#>  0.5213  0.9530 -0.4761 -0.6401
#> -0.2843 -0.8601 -0.7555  1.0457
#> -0.3219  0.4671  2.0144  3.1305
#> -0.7244 -0.9120  2.1248  1.2126
#> -0.7866  1.5632 -1.4581 -1.6521
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.8844
#> -0.0964
#> -0.6067
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.5198
#>  1.2186
#>  1.4260
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.1124
#>  0.6556
#>  2.6156
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.3865
#>  0.2369
#>  0.8536
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.2660
#>  0.7218
#> -0.9317
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.3408
#>  1.3695
#> -0.6714
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.8833
#>  1.6252
#>  0.3801
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.5000
#> -0.3880
#>  1.3117
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.5339
#>  0.8954
#> -1.7333
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.0540
#>  0.1057
#>  1.5848
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.2711
#> -0.8584
#>  1.1657
#>  0.0525
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.9575
#> -0.8781
#> -1.8560
#>  0.1529
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.0188
#>  0.1271
#>  0.7161
#>  0.8354
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.4195
#>  0.0223
#>  2.7435
#>  0.0151
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.0067
#> -2.3031
#> -1.9115
#> -1.4543
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.5213
#>  0.9530
#> -0.4761
#> -0.6401
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.2843
#> -0.8601
#> -0.7555
#>  1.0457
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.3219
#>  0.4671
#>  2.0144
#>  3.1305
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.7244
#> -0.9120
#>  2.1248
#>  1.2126
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.7866
#>  1.5632
#> -1.4581
#> -1.6521
#> [ CPUFloatType{4} ]
#> 
#> 
```
