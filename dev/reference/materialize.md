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

  (`character(1)` or
  [`environment()`](https://rdrr.io/r/base/environment.html) or
  `NULL`)  
  Optional cache for (intermediate) materialization results. Per
  default, caching will be enabled when the same dataset or data
  descriptor (with different output pointer) is used for more than one
  lazy tensor column.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)s
or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md))

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

For this reason it is possible to provide a cache environment. The hash
key for a) is the hash of the indices and the dataset. The hash key for
b) is the hash of the indices, dataset and preprocessing graph.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#>  0.6176  1.7595 -1.4461
#> -0.2300 -0.3446  2.4462
#> -1.1101  0.2114 -1.3817
#>  0.5695 -0.7719  0.8068
#>  2.1724 -0.0845  0.9050
#>  0.6967  1.8504  1.1373
#> -0.0343  1.0635 -1.0245
#>  0.8369  0.6592  0.4434
#>  0.0317 -1.6023 -1.6960
#>  1.0297  0.1046 -0.3374
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.6176
#>  1.7595
#> -1.4461
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.2300
#> -0.3446
#>  2.4462
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.1101
#>  0.2114
#> -1.3817
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.5695
#> -0.7719
#>  0.8068
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  2.1724
#> -0.0845
#>  0.9050
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.6967
#>  1.8504
#>  1.1373
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.0343
#>  1.0635
#> -1.0245
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.8369
#>  0.6592
#>  0.4434
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.0317
#> -1.6023
#> -1.6960
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.0297
#>  0.1046
#> -0.3374
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.6176  1.7595 -1.4461
#> -0.2300 -0.3446  2.4462
#> -1.1101  0.2114 -1.3817
#>  0.5695 -0.7719  0.8068
#>  2.1724 -0.0845  0.9050
#>  0.6967  1.8504  1.1373
#> -0.0343  1.0635 -1.0245
#>  0.8369  0.6592  0.4434
#>  0.0317 -1.6023 -1.6960
#>  1.0297  0.1046 -0.3374
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.2483  1.0031  0.2318  1.0030
#>  1.6688 -1.2597 -0.1584  2.4091
#>  1.0775  2.5254 -1.6415 -0.6493
#>  0.2442  0.9642  1.9452 -2.3286
#>  0.5273 -1.5816 -1.1672  0.0470
#> -2.2687  0.4961 -0.6268 -0.7501
#> -0.3102  0.9533 -0.6399  0.0925
#>  0.3217 -0.9364  0.1814  0.0211
#> -0.9146  0.5021 -1.2352  0.6611
#> -0.9745 -0.6580 -0.5246 -0.8032
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.6176
#>  1.7595
#> -1.4461
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.2300
#> -0.3446
#>  2.4462
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.1101
#>  0.2114
#> -1.3817
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.5695
#> -0.7719
#>  0.8068
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  2.1724
#> -0.0845
#>  0.9050
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.6967
#>  1.8504
#>  1.1373
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.0343
#>  1.0635
#> -1.0245
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.8369
#>  0.6592
#>  0.4434
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.0317
#> -1.6023
#> -1.6960
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.0297
#>  0.1046
#> -0.3374
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.2483
#>  1.0031
#>  0.2318
#>  1.0030
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.6688
#> -1.2597
#> -0.1584
#>  2.4091
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.0775
#>  2.5254
#> -1.6415
#> -0.6493
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2442
#>  0.9642
#>  1.9452
#> -2.3286
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.5273
#> -1.5816
#> -1.1672
#>  0.0470
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -2.2687
#>  0.4961
#> -0.6268
#> -0.7501
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.3102
#>  0.9533
#> -0.6399
#>  0.0925
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.3217
#> -0.9364
#>  0.1814
#>  0.0211
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.9146
#>  0.5021
#> -1.2352
#>  0.6611
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.9745
#> -0.6580
#> -0.5246
#> -0.8032
#> [ CPUFloatType{4} ]
#> 
#> 
```
