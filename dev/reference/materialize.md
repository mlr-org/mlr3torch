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
#>  1.4304  0.8421 -0.9483
#>  0.9708  1.0412 -0.2172
#> -1.1674  1.0601 -0.0952
#> -0.8505  0.5049  0.9011
#>  0.6973 -0.9326 -0.6945
#>  1.1459  0.1438  1.9412
#> -0.0655  0.8794 -0.3332
#>  0.7908 -0.1809  0.2291
#>  0.8047 -0.1151  0.4530
#> -0.0199 -0.6995  0.0707
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.4304
#>  0.8421
#> -0.9483
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.9708
#>  1.0412
#> -0.2172
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.1674
#>  1.0601
#> -0.0952
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.8505
#>  0.5049
#>  0.9011
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.6973
#> -0.9326
#> -0.6945
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.1459
#>  0.1438
#>  1.9412
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.0655
#>  0.8794
#> -0.3332
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.7908
#> -0.1809
#>  0.2291
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.8047
#> -0.1151
#>  0.4530
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.0199
#> -0.6995
#>  0.0707
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.4304  0.8421 -0.9483
#>  0.9708  1.0412 -0.2172
#> -1.1674  1.0601 -0.0952
#> -0.8505  0.5049  0.9011
#>  0.6973 -0.9326 -0.6945
#>  1.1459  0.1438  1.9412
#> -0.0655  0.8794 -0.3332
#>  0.7908 -0.1809  0.2291
#>  0.8047 -0.1151  0.4530
#> -0.0199 -0.6995  0.0707
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0505  0.5365 -2.1787  0.8745
#> -0.2107  0.7065 -0.1733  0.0924
#>  0.4602  1.6715 -0.9562 -0.3363
#> -1.2332  0.5869  0.6474 -0.1416
#> -0.3057  0.9372  1.8154 -0.5082
#>  0.3440 -0.4655 -1.4389  0.8383
#> -0.6307  1.5177  0.9533 -2.0458
#>  0.0621 -1.0646  1.1602 -0.6205
#> -1.0884  0.4360 -2.1702  0.6610
#>  0.9378  0.7833  0.9666  0.0919
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.4304
#>  0.8421
#> -0.9483
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.9708
#>  1.0412
#> -0.2172
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.1674
#>  1.0601
#> -0.0952
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.8505
#>  0.5049
#>  0.9011
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.6973
#> -0.9326
#> -0.6945
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.1459
#>  0.1438
#>  1.9412
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.0655
#>  0.8794
#> -0.3332
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.7908
#> -0.1809
#>  0.2291
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.8047
#> -0.1151
#>  0.4530
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.0199
#> -0.6995
#>  0.0707
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0505
#>  0.5365
#> -2.1787
#>  0.8745
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.2107
#>  0.7065
#> -0.1733
#>  0.0924
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.4602
#>  1.6715
#> -0.9562
#> -0.3363
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.2332
#>  0.5869
#>  0.6474
#> -0.1416
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.3057
#>  0.9372
#>  1.8154
#> -0.5082
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.3440
#> -0.4655
#> -1.4389
#>  0.8383
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.6307
#>  1.5177
#>  0.9533
#> -2.0458
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.0621
#> -1.0646
#>  1.1602
#> -0.6205
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.0884
#>  0.4360
#> -2.1702
#>  0.6610
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.9378
#>  0.7833
#>  0.9666
#>  0.0919
#> [ CPUFloatType{4} ]
#> 
#> 
```
