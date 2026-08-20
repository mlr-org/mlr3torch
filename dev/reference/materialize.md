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
#> -1.2066 -0.9711 -0.6252
#> -1.0544  0.8607 -1.0710
#>  0.0025 -0.7921  0.9101
#>  0.7639  0.7213  0.8944
#>  0.7901  0.3485 -0.7249
#> -0.0618 -1.5513 -0.2635
#>  0.0096 -0.1156  0.2710
#>  1.1526  0.8212  0.6370
#> -0.3015  0.4774  2.8312
#>  0.4623 -0.9768 -0.4166
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.2066
#> -0.9711
#> -0.6252
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.0544
#>  0.8607
#> -1.0710
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.0025
#> -0.7921
#>  0.9101
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7639
#>  0.7213
#>  0.8944
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.7901
#>  0.3485
#> -0.7249
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.0618
#> -1.5513
#> -0.2635
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.0096
#> -0.1156
#>  0.2710
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.1526
#>  0.8212
#>  0.6370
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.3015
#>  0.4774
#>  2.8312
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4623
#> -0.9768
#> -0.4166
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.2066 -0.9711 -0.6252
#> -1.0544  0.8607 -1.0710
#>  0.0025 -0.7921  0.9101
#>  0.7639  0.7213  0.8944
#>  0.7901  0.3485 -0.7249
#> -0.0618 -1.5513 -0.2635
#>  0.0096 -0.1156  0.2710
#>  1.1526  0.8212  0.6370
#> -0.3015  0.4774  2.8312
#>  0.4623 -0.9768 -0.4166
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.9118  0.2111 -0.2325 -0.2464
#>  0.9585 -0.1222  0.6144 -1.8448
#>  0.7066  0.6967  0.8471 -2.9350
#>  0.3881  1.2810  1.1009  0.3375
#>  0.6314  0.0926  0.7311  0.2736
#>  0.8253 -1.1030  1.1290 -1.0529
#>  0.7494 -0.5712  0.4586  0.4165
#>  0.4703  0.1141  0.9875 -0.9971
#> -1.3627 -0.4790  0.2546 -0.2352
#> -0.8639 -0.4018 -1.2538 -0.9523
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.2066
#> -0.9711
#> -0.6252
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.0544
#>  0.8607
#> -1.0710
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.0025
#> -0.7921
#>  0.9101
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7639
#>  0.7213
#>  0.8944
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.7901
#>  0.3485
#> -0.7249
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.0618
#> -1.5513
#> -0.2635
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.0096
#> -0.1156
#>  0.2710
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.1526
#>  0.8212
#>  0.6370
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.3015
#>  0.4774
#>  2.8312
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4623
#> -0.9768
#> -0.4166
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.9118
#>  0.2111
#> -0.2325
#> -0.2464
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.9585
#> -0.1222
#>  0.6144
#> -1.8448
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.7066
#>  0.6967
#>  0.8471
#> -2.9350
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.3881
#>  1.2810
#>  1.1009
#>  0.3375
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.6314
#>  0.0926
#>  0.7311
#>  0.2736
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.8253
#> -1.1030
#>  1.1290
#> -1.0529
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.7494
#> -0.5712
#>  0.4586
#>  0.4165
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4703
#>  0.1141
#>  0.9875
#> -0.9971
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.3627
#> -0.4790
#>  0.2546
#> -0.2352
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.8639
#> -0.4018
#> -1.2538
#> -0.9523
#> [ CPUFloatType{4} ]
#> 
#> 
```
