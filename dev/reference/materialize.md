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
#> -0.6496  0.2276  0.5999
#>  1.5496 -1.0939 -0.1664
#> -0.4068 -1.1112  0.0081
#> -1.3282  1.2201 -1.2404
#> -1.5184 -1.5921  1.2673
#> -0.4037  1.0982 -1.1416
#>  0.6872  0.9651 -1.1538
#>  0.1132  2.4986  1.8734
#> -2.0698  1.4794  1.1300
#> -1.6771 -1.2896 -0.0491
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.6496
#>  0.2276
#>  0.5999
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.5496
#> -1.0939
#> -0.1664
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.4068
#> -1.1112
#>  0.0081
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.3282
#>  1.2201
#> -1.2404
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.5184
#> -1.5921
#>  1.2673
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.4037
#>  1.0982
#> -1.1416
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.6872
#>  0.9651
#> -1.1538
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.1132
#>  2.4986
#>  1.8734
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -2.0698
#>  1.4794
#>  1.1300
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.6771
#> -1.2896
#> -0.0491
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.6496  0.2276  0.5999
#>  1.5496 -1.0939 -0.1664
#> -0.4068 -1.1112  0.0081
#> -1.3282  1.2201 -1.2404
#> -1.5184 -1.5921  1.2673
#> -0.4037  1.0982 -1.1416
#>  0.6872  0.9651 -1.1538
#>  0.1132  2.4986  1.8734
#> -2.0698  1.4794  1.1300
#> -1.6771 -1.2896 -0.0491
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.2077  1.1680  1.0827 -0.2761
#>  0.8783  0.3482  1.7669  0.6537
#> -0.2523  0.4290 -0.2200 -0.3877
#>  0.1924  0.0588 -0.1923 -0.2567
#>  2.8268  0.6947 -0.4789  0.5679
#>  0.1578  0.4730 -0.4746 -0.4492
#>  0.0856  0.6254  0.0022  1.4035
#> -0.6532 -0.0281  1.6259 -0.9494
#> -0.7287  0.2025  0.4759 -0.9455
#> -0.2066  1.0029 -1.0235  1.9718
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.6496
#>  0.2276
#>  0.5999
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.5496
#> -1.0939
#> -0.1664
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.4068
#> -1.1112
#>  0.0081
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.3282
#>  1.2201
#> -1.2404
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.5184
#> -1.5921
#>  1.2673
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.4037
#>  1.0982
#> -1.1416
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.6872
#>  0.9651
#> -1.1538
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.1132
#>  2.4986
#>  1.8734
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -2.0698
#>  1.4794
#>  1.1300
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.6771
#> -1.2896
#> -0.0491
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.2077
#>  1.1680
#>  1.0827
#> -0.2761
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.8783
#>  0.3482
#>  1.7669
#>  0.6537
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.2523
#>  0.4290
#> -0.2200
#> -0.3877
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.1924
#>  0.0588
#> -0.1923
#> -0.2567
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  2.8268
#>  0.6947
#> -0.4789
#>  0.5679
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.1578
#>  0.4730
#> -0.4746
#> -0.4492
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.0856
#>  0.6254
#>  0.0022
#>  1.4035
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.6532
#> -0.0281
#>  1.6259
#> -0.9494
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.7287
#>  0.2025
#>  0.4759
#> -0.9455
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.2066
#>  1.0029
#> -1.0235
#>  1.9718
#> [ CPUFloatType{4} ]
#> 
#> 
```
