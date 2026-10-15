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
#>  0.8413  1.1281 -0.2613
#>  0.8226 -0.0009 -0.7427
#> -0.8637 -0.3477  0.2707
#> -0.5627  0.4097  1.1816
#> -0.4841 -0.9939  0.0046
#>  1.1341  0.1631  0.2720
#> -0.2182 -0.2415  0.0477
#> -0.8838 -0.5962 -1.8638
#>  1.3354 -1.1398 -0.1476
#>  0.2965 -0.3740  2.4192
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.8413
#>  1.1281
#> -0.2613
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.8226
#> -0.0009
#> -0.7427
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.8637
#> -0.3477
#>  0.2707
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.5627
#>  0.4097
#>  1.1816
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.4841
#> -0.9939
#>  0.0046
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.1341
#>  0.1631
#>  0.2720
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.2182
#> -0.2415
#>  0.0477
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.8838
#> -0.5962
#> -1.8638
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.3354
#> -1.1398
#> -0.1476
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.2965
#> -0.3740
#>  2.4192
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.8413  1.1281 -0.2613
#>  0.8226 -0.0009 -0.7427
#> -0.8637 -0.3477  0.2707
#> -0.5627  0.4097  1.1816
#> -0.4841 -0.9939  0.0046
#>  1.1341  0.1631  0.2720
#> -0.2182 -0.2415  0.0477
#> -0.8838 -0.5962 -1.8638
#>  1.3354 -1.1398 -0.1476
#>  0.2965 -0.3740  2.4192
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.5800  0.4187  0.2520 -1.8237
#>  0.4587  1.5642 -0.2721 -0.8951
#>  0.9271 -0.6366  0.3420  0.9365
#>  0.2696 -0.6125  0.7286  0.1336
#>  1.9607 -0.2912  2.1105  1.6492
#>  2.6181 -0.3461  0.1371 -0.3870
#> -0.3445 -2.7665 -0.4491 -0.6289
#>  0.4234  0.5463  0.4232 -0.4684
#> -1.8369  1.0368  2.1768  1.3550
#>  0.4780 -1.2826 -0.2504 -0.9594
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.8413
#>  1.1281
#> -0.2613
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.8226
#> -0.0009
#> -0.7427
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.8637
#> -0.3477
#>  0.2707
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.5627
#>  0.4097
#>  1.1816
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.4841
#> -0.9939
#>  0.0046
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.1341
#>  0.1631
#>  0.2720
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.2182
#> -0.2415
#>  0.0477
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.8838
#> -0.5962
#> -1.8638
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.3354
#> -1.1398
#> -0.1476
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.2965
#> -0.3740
#>  2.4192
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.5800
#>  0.4187
#>  0.2520
#> -1.8237
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.4587
#>  1.5642
#> -0.2721
#> -0.8951
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.9271
#> -0.6366
#>  0.3420
#>  0.9365
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2696
#> -0.6125
#>  0.7286
#>  0.1336
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.9607
#> -0.2912
#>  2.1105
#>  1.6492
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  2.6181
#> -0.3461
#>  0.1371
#> -0.3870
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.3445
#> -2.7665
#> -0.4491
#> -0.6289
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4234
#>  0.5463
#>  0.4232
#> -0.4684
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.8369
#>  1.0368
#>  2.1768
#>  1.3550
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.4780
#> -1.2826
#> -0.2504
#> -0.9594
#> [ CPUFloatType{4} ]
#> 
#> 
```
