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
#> -1.5169  0.8511 -0.0041
#>  1.5816 -0.1057 -2.1176
#> -1.4020 -1.6496 -1.2790
#> -0.6425  0.8694 -1.6747
#> -1.2283  0.1623  1.9773
#> -1.4449  1.2465 -1.4594
#>  0.0258  0.7850 -0.3035
#>  0.6878  0.9628  2.2898
#>  0.3870  0.3633  1.1638
#>  0.3425 -0.0578  2.1943
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.5169
#>  0.8511
#> -0.0041
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.5816
#> -0.1057
#> -2.1176
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.4020
#> -1.6496
#> -1.2790
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6425
#>  0.8694
#> -1.6747
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.2283
#>  0.1623
#>  1.9773
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -1.4449
#>  1.2465
#> -1.4594
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.0258
#>  0.7850
#> -0.3035
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.6878
#>  0.9628
#>  2.2898
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.3870
#>  0.3633
#>  1.1638
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.3425
#> -0.0578
#>  2.1943
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.5169  0.8511 -0.0041
#>  1.5816 -0.1057 -2.1176
#> -1.4020 -1.6496 -1.2790
#> -0.6425  0.8694 -1.6747
#> -1.2283  0.1623  1.9773
#> -1.4449  1.2465 -1.4594
#>  0.0258  0.7850 -0.3035
#>  0.6878  0.9628  2.2898
#>  0.3870  0.3633  1.1638
#>  0.3425 -0.0578  2.1943
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.1505 -1.1492  1.3154  0.0091
#>  1.3015  0.2540 -0.1379  0.3797
#>  1.3240  2.2495  0.0409 -0.4944
#> -0.6177 -0.1370 -1.2322  1.5382
#>  0.0400  0.0814  0.1458 -0.1317
#> -1.4074  1.0259 -0.5117  0.4370
#> -0.1915  0.3072 -0.4018  0.9647
#>  0.7613 -0.3053 -1.4355  0.0078
#>  0.0225 -0.9724  0.2467 -0.4617
#> -0.3848 -0.5826  0.1998 -1.4592
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.5169
#>  0.8511
#> -0.0041
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.5816
#> -0.1057
#> -2.1176
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.4020
#> -1.6496
#> -1.2790
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6425
#>  0.8694
#> -1.6747
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.2283
#>  0.1623
#>  1.9773
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -1.4449
#>  1.2465
#> -1.4594
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.0258
#>  0.7850
#> -0.3035
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.6878
#>  0.9628
#>  2.2898
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.3870
#>  0.3633
#>  1.1638
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.3425
#> -0.0578
#>  2.1943
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.1505
#> -1.1492
#>  1.3154
#>  0.0091
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.3015
#>  0.2540
#> -0.1379
#>  0.3797
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.3240
#>  2.2495
#>  0.0409
#> -0.4944
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.6177
#> -0.1370
#> -1.2322
#>  1.5382
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.0400
#>  0.0814
#>  0.1458
#> -0.1317
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.4074
#>  1.0259
#> -0.5117
#>  0.4370
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.1915
#>  0.3072
#> -0.4018
#>  0.9647
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.7613
#> -0.3053
#> -1.4355
#>  0.0078
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.0225
#> -0.9724
#>  0.2467
#> -0.4617
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.3848
#> -0.5826
#>  0.1998
#> -1.4592
#> [ CPUFloatType{4} ]
#> 
#> 
```
