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
#> -1.2230  0.5530  0.8729
#> -0.1630  0.7172 -0.0639
#>  1.8528 -2.3120 -1.0875
#>  0.9788 -0.5613 -0.3742
#> -0.3182 -0.1488 -1.1348
#> -0.4144 -1.6900 -1.1038
#>  0.0570  1.1202 -0.5477
#> -1.1722 -0.8676 -0.5982
#> -1.3749 -0.0903 -1.5176
#>  0.4691 -0.3354 -0.0707
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.2230
#>  0.5530
#>  0.8729
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.1630
#>  0.7172
#> -0.0639
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.8528
#> -2.3120
#> -1.0875
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.9788
#> -0.5613
#> -0.3742
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.3182
#> -0.1488
#> -1.1348
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.4144
#> -1.6900
#> -1.1038
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.0570
#>  1.1202
#> -0.5477
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.1722
#> -0.8676
#> -0.5982
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.3749
#> -0.0903
#> -1.5176
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4691
#> -0.3354
#> -0.0707
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.2230  0.5530  0.8729
#> -0.1630  0.7172 -0.0639
#>  1.8528 -2.3120 -1.0875
#>  0.9788 -0.5613 -0.3742
#> -0.3182 -0.1488 -1.1348
#> -0.4144 -1.6900 -1.1038
#>  0.0570  1.1202 -0.5477
#> -1.1722 -0.8676 -0.5982
#> -1.3749 -0.0903 -1.5176
#>  0.4691 -0.3354 -0.0707
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0874  0.7154  0.4111 -0.9928
#> -0.5060 -1.0360 -0.5367  0.0051
#> -1.0292 -0.4475 -0.6285  0.3841
#> -0.2187 -0.8686  0.0007  0.0419
#>  1.3075 -0.4823  0.4443 -0.6720
#>  0.4370  0.6987  0.0090  0.3367
#> -1.0517 -0.0283 -0.4812 -0.3443
#> -1.1249  0.7734  0.6586 -1.5538
#>  1.0963  0.3845  1.0660 -1.3995
#> -0.3911 -1.1876  0.8652  0.0862
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.2230
#>  0.5530
#>  0.8729
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.1630
#>  0.7172
#> -0.0639
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.8528
#> -2.3120
#> -1.0875
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.9788
#> -0.5613
#> -0.3742
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.3182
#> -0.1488
#> -1.1348
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.4144
#> -1.6900
#> -1.1038
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.0570
#>  1.1202
#> -0.5477
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.1722
#> -0.8676
#> -0.5982
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.3749
#> -0.0903
#> -1.5176
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4691
#> -0.3354
#> -0.0707
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0874
#>  0.7154
#>  0.4111
#> -0.9928
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.5060
#> -1.0360
#> -0.5367
#>  0.0051
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.0292
#> -0.4475
#> -0.6285
#>  0.3841
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.2187
#> -0.8686
#>  0.0007
#>  0.0419
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.3075
#> -0.4823
#>  0.4443
#> -0.6720
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4370
#>  0.6987
#>  0.0090
#>  0.3367
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.0517
#> -0.0283
#> -0.4812
#> -0.3443
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.1249
#>  0.7734
#>  0.6586
#> -1.5538
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.0963
#>  0.3845
#>  1.0660
#> -1.3995
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.3911
#> -1.1876
#>  0.8652
#>  0.0862
#> [ CPUFloatType{4} ]
#> 
#> 
```
