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
#> -1.6468 -0.4010  0.0707
#>  1.2708  0.3998 -0.3664
#> -1.4953  0.6135  0.3640
#>  0.2385  0.3816 -1.0356
#> -0.3650 -0.4819  0.2326
#>  0.4332  0.0359 -0.0961
#>  0.1688  1.4428 -0.9584
#> -2.3409  1.0252 -0.1569
#> -1.3609  1.1974 -0.5460
#>  0.7989  1.6167 -0.3145
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.6468
#> -0.4010
#>  0.0707
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.2708
#>  0.3998
#> -0.3664
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.4953
#>  0.6135
#>  0.3640
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.2385
#>  0.3816
#> -1.0356
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.3650
#> -0.4819
#>  0.2326
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.4332
#>  0.0359
#> -0.0961
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.1688
#>  1.4428
#> -0.9584
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -2.3409
#>  1.0252
#> -0.1569
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.3609
#>  1.1974
#> -0.5460
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.7989
#>  1.6167
#> -0.3145
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.6468 -0.4010  0.0707
#>  1.2708  0.3998 -0.3664
#> -1.4953  0.6135  0.3640
#>  0.2385  0.3816 -1.0356
#> -0.3650 -0.4819  0.2326
#>  0.4332  0.0359 -0.0961
#>  0.1688  1.4428 -0.9584
#> -2.3409  1.0252 -0.1569
#> -1.3609  1.1974 -0.5460
#>  0.7989  1.6167 -0.3145
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.9004 -0.3699 -0.4767 -1.2311
#> -2.2810 -0.6537 -0.4616 -0.1916
#>  1.8099 -0.2131  0.1804 -0.1325
#>  0.0769 -1.3542  0.8906  0.0534
#> -1.2877  0.7140  0.4491  1.3225
#>  0.3695 -1.3909 -1.3588 -1.4732
#>  0.0724 -0.0540  0.0247  2.4824
#>  1.6874  0.7442 -0.7002 -1.3477
#> -0.2157  0.3025  1.1162 -1.2669
#>  2.0979 -0.9109  0.1995 -1.3441
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.6468
#> -0.4010
#>  0.0707
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.2708
#>  0.3998
#> -0.3664
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.4953
#>  0.6135
#>  0.3640
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.2385
#>  0.3816
#> -1.0356
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.3650
#> -0.4819
#>  0.2326
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.4332
#>  0.0359
#> -0.0961
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.1688
#>  1.4428
#> -0.9584
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -2.3409
#>  1.0252
#> -0.1569
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.3609
#>  1.1974
#> -0.5460
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.7989
#>  1.6167
#> -0.3145
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.9004
#> -0.3699
#> -0.4767
#> -1.2311
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -2.2810
#> -0.6537
#> -0.4616
#> -0.1916
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.8099
#> -0.2131
#>  0.1804
#> -0.1325
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.0769
#> -1.3542
#>  0.8906
#>  0.0534
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.2877
#>  0.7140
#>  0.4491
#>  1.3225
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.3695
#> -1.3909
#> -1.3588
#> -1.4732
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.0724
#> -0.0540
#>  0.0247
#>  2.4824
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.6874
#>  0.7442
#> -0.7002
#> -1.3477
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.2157
#>  0.3025
#>  1.1162
#> -1.2669
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  2.0979
#> -0.9109
#>  0.1995
#> -1.3441
#> [ CPUFloatType{4} ]
#> 
#> 
```
