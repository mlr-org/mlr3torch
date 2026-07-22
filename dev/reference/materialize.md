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
#>  0.3729 -0.4731 -0.5995
#> -1.1427  0.0048  0.4111
#>  0.4695 -0.3509  1.2005
#> -0.0755  0.3016 -0.0081
#>  1.9594 -1.2044  0.9286
#> -0.0360 -0.1935  0.1653
#> -0.9874  0.1361 -0.0319
#>  0.4320  0.8713  0.4814
#>  1.2076 -0.6025  0.0751
#> -1.4588 -1.0766  0.4541
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3729
#> -0.4731
#> -0.5995
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.1427
#>  0.0048
#>  0.4111
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.4695
#> -0.3509
#>  1.2005
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.0755
#>  0.3016
#> -0.0081
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.9594
#> -1.2044
#>  0.9286
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.0360
#> -0.1935
#>  0.1653
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.9874
#>  0.1361
#> -0.0319
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.4320
#>  0.8713
#>  0.4814
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.2076
#> -0.6025
#>  0.0751
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.4588
#> -1.0766
#>  0.4541
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3729 -0.4731 -0.5995
#> -1.1427  0.0048  0.4111
#>  0.4695 -0.3509  1.2005
#> -0.0755  0.3016 -0.0081
#>  1.9594 -1.2044  0.9286
#> -0.0360 -0.1935  0.1653
#> -0.9874  0.1361 -0.0319
#>  0.4320  0.8713  0.4814
#>  1.2076 -0.6025  0.0751
#> -1.4588 -1.0766  0.4541
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.4686  0.0218  1.0796  0.7797
#>  0.9212  0.3336  0.6026  1.6168
#> -1.0164  0.8445 -0.4496  1.2037
#> -1.2567 -0.6353  0.2212  0.7166
#> -0.9215 -0.1328 -1.1439 -0.4685
#>  0.5648  1.6028  0.4527  1.2786
#> -0.5775  0.3611  2.3440 -0.5904
#>  2.3313 -1.5321 -0.0511  0.6694
#> -1.1153 -1.1399 -2.1264  0.2093
#>  1.2932 -0.8268  1.5101  0.8582
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3729
#> -0.4731
#> -0.5995
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.1427
#>  0.0048
#>  0.4111
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.4695
#> -0.3509
#>  1.2005
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.0755
#>  0.3016
#> -0.0081
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.9594
#> -1.2044
#>  0.9286
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.0360
#> -0.1935
#>  0.1653
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.9874
#>  0.1361
#> -0.0319
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.4320
#>  0.8713
#>  0.4814
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.2076
#> -0.6025
#>  0.0751
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.4588
#> -1.0766
#>  0.4541
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.4686
#>  0.0218
#>  1.0796
#>  0.7797
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.9212
#>  0.3336
#>  0.6026
#>  1.6168
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.0164
#>  0.8445
#> -0.4496
#>  1.2037
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.2567
#> -0.6353
#>  0.2212
#>  0.7166
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.9215
#> -0.1328
#> -1.1439
#> -0.4685
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.5648
#>  1.6028
#>  0.4527
#>  1.2786
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.5775
#>  0.3611
#>  2.3440
#> -0.5904
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  2.3313
#> -1.5321
#> -0.0511
#>  0.6694
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.1153
#> -1.1399
#> -2.1264
#>  0.2093
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.2932
#> -0.8268
#>  1.5101
#>  0.8582
#> [ CPUFloatType{4} ]
#> 
#> 
```
