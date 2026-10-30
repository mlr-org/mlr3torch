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
#> -1.1205 -0.4638  0.6175
#> -1.5196 -0.8895 -1.0000
#> -0.6180  0.0450 -2.2002
#>  0.4112 -0.2235  0.4153
#>  1.3911  1.5860 -0.8347
#>  1.3926 -0.7085 -1.0263
#> -0.7477 -0.2287  1.1087
#> -0.0901 -0.3499  0.0613
#> -0.5371  0.2169 -0.7428
#> -0.6416 -1.3736 -1.0774
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.1205
#> -0.4638
#>  0.6175
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.5196
#> -0.8895
#> -1.0000
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.6180
#>  0.0450
#> -2.2002
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.4112
#> -0.2235
#>  0.4153
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.3911
#>  1.5860
#> -0.8347
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.3926
#> -0.7085
#> -1.0263
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.7477
#> -0.2287
#>  1.1087
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.0901
#> -0.3499
#>  0.0613
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.5371
#>  0.2169
#> -0.7428
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.6416
#> -1.3736
#> -1.0774
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.1205 -0.4638  0.6175
#> -1.5196 -0.8895 -1.0000
#> -0.6180  0.0450 -2.2002
#>  0.4112 -0.2235  0.4153
#>  1.3911  1.5860 -0.8347
#>  1.3926 -0.7085 -1.0263
#> -0.7477 -0.2287  1.1087
#> -0.0901 -0.3499  0.0613
#> -0.5371  0.2169 -0.7428
#> -0.6416 -1.3736 -1.0774
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.8947  0.4270  0.6815 -1.2334
#> -0.0425 -0.7314  0.3840  0.6041
#> -0.5685 -1.6927  0.1225 -0.2554
#> -0.5061 -0.9451  1.2419  0.7277
#> -1.2785 -0.8408 -0.5077 -0.1939
#>  0.8063 -2.0943 -0.7155  0.8505
#>  0.2574  1.2776 -0.9966  1.2897
#> -1.3482 -0.1067  0.8194 -0.0129
#> -1.0670  1.0263 -0.9809  0.6165
#>  0.0676 -0.9294  0.5544  0.1873
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.1205
#> -0.4638
#>  0.6175
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.5196
#> -0.8895
#> -1.0000
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.6180
#>  0.0450
#> -2.2002
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.4112
#> -0.2235
#>  0.4153
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.3911
#>  1.5860
#> -0.8347
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.3926
#> -0.7085
#> -1.0263
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.7477
#> -0.2287
#>  1.1087
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.0901
#> -0.3499
#>  0.0613
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.5371
#>  0.2169
#> -0.7428
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.6416
#> -1.3736
#> -1.0774
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.8947
#>  0.4270
#>  0.6815
#> -1.2334
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.0425
#> -0.7314
#>  0.3840
#>  0.6041
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.5685
#> -1.6927
#>  0.1225
#> -0.2554
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.5061
#> -0.9451
#>  1.2419
#>  0.7277
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.2785
#> -0.8408
#> -0.5077
#> -0.1939
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.8063
#> -2.0943
#> -0.7155
#>  0.8505
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.2574
#>  1.2776
#> -0.9966
#>  1.2897
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.3482
#> -0.1067
#>  0.8194
#> -0.0129
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.0670
#>  1.0263
#> -0.9809
#>  0.6165
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.0676
#> -0.9294
#>  0.5544
#>  0.1873
#> [ CPUFloatType{4} ]
#> 
#> 
```
