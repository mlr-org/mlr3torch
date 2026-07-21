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
#> -0.8280 -0.6815  0.5912
#> -0.0212  0.0085  0.0655
#>  0.5409  0.1904 -0.2779
#>  0.3240 -0.5136  0.4077
#>  0.6876 -0.2582  0.1021
#>  0.0299 -0.8524 -0.6121
#>  0.5235 -1.7484  0.0272
#> -0.4338 -0.2467  1.2959
#> -0.7284 -1.7474  0.4666
#> -0.0060  0.1941 -1.6556
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.8280
#> -0.6815
#>  0.5912
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> 0.01 *
#> -2.1160
#>  0.8486
#>  6.5464
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.5409
#>  0.1904
#> -0.2779
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.3240
#> -0.5136
#>  0.4077
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.6876
#> -0.2582
#>  0.1021
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.0299
#> -0.8524
#> -0.6121
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.5235
#> -1.7484
#>  0.0272
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.4338
#> -0.2467
#>  1.2959
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.7284
#> -1.7474
#>  0.4666
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.0060
#>  0.1941
#> -1.6556
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.8280 -0.6815  0.5912
#> -0.0212  0.0085  0.0655
#>  0.5409  0.1904 -0.2779
#>  0.3240 -0.5136  0.4077
#>  0.6876 -0.2582  0.1021
#>  0.0299 -0.8524 -0.6121
#>  0.5235 -1.7484  0.0272
#> -0.4338 -0.2467  1.2959
#> -0.7284 -1.7474  0.4666
#> -0.0060  0.1941 -1.6556
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.3276  0.2633 -0.0131 -0.0263
#> -1.1338 -0.5476  3.0648 -1.1508
#>  1.4143 -0.9657  0.5119 -0.2914
#> -2.3869 -0.2377 -0.2510 -0.4923
#>  2.2140 -0.5112 -0.5551  1.5439
#> -0.2261  0.4606  1.3479  0.5787
#>  0.0306 -0.6682  1.6115  0.8577
#>  0.4693  0.3171 -1.3747 -1.1342
#> -0.6874  0.5965  0.6708  0.2988
#> -0.4526 -0.1067  0.4173  1.4614
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.8280
#> -0.6815
#>  0.5912
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> 0.01 *
#> -2.1160
#>  0.8486
#>  6.5464
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.5409
#>  0.1904
#> -0.2779
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.3240
#> -0.5136
#>  0.4077
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.6876
#> -0.2582
#>  0.1021
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.0299
#> -0.8524
#> -0.6121
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.5235
#> -1.7484
#>  0.0272
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.4338
#> -0.2467
#>  1.2959
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.7284
#> -1.7474
#>  0.4666
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.0060
#>  0.1941
#> -1.6556
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.3276
#>  0.2633
#> -0.0131
#> -0.0263
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.1338
#> -0.5476
#>  3.0648
#> -1.1508
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.4143
#> -0.9657
#>  0.5119
#> -0.2914
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -2.3869
#> -0.2377
#> -0.2510
#> -0.4923
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  2.2140
#> -0.5112
#> -0.5551
#>  1.5439
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.2261
#>  0.4606
#>  1.3479
#>  0.5787
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.0306
#> -0.6682
#>  1.6115
#>  0.8577
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4693
#>  0.3171
#> -1.3747
#> -1.1342
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.6874
#>  0.5965
#>  0.6708
#>  0.2988
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.4526
#> -0.1067
#>  0.4173
#>  1.4614
#> [ CPUFloatType{4} ]
#> 
#> 
```
