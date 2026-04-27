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
#> -0.0995  0.0075 -0.0831
#>  1.4152  1.0710 -0.8902
#> -0.4422  0.1899  0.1474
#>  0.8595 -2.0373  0.2836
#>  1.6798 -1.1397  0.8287
#> -0.9204  3.2267  0.5465
#>  0.4931 -1.4769  0.4412
#> -0.1975  1.0870  1.4432
#> -0.3773 -0.4296  0.2689
#> -0.4434 -2.1599  1.4137
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> 0.01 *
#> -9.9543
#>  0.7549
#> -8.3099
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.4152
#>  1.0710
#> -0.8902
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.4422
#>  0.1899
#>  0.1474
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.8595
#> -2.0373
#>  0.2836
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.6798
#> -1.1397
#>  0.8287
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.9204
#>  3.2267
#>  0.5465
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.4931
#> -1.4769
#>  0.4412
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.1975
#>  1.0870
#>  1.4432
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.3773
#> -0.4296
#>  0.2689
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.4434
#> -2.1599
#>  1.4137
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.0995  0.0075 -0.0831
#>  1.4152  1.0710 -0.8902
#> -0.4422  0.1899  0.1474
#>  0.8595 -2.0373  0.2836
#>  1.6798 -1.1397  0.8287
#> -0.9204  3.2267  0.5465
#>  0.4931 -1.4769  0.4412
#> -0.1975  1.0870  1.4432
#> -0.3773 -0.4296  0.2689
#> -0.4434 -2.1599  1.4137
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.2703 -0.0146  1.4458 -0.6430
#>  0.0490 -0.8759 -0.3446  0.7235
#> -1.2982  0.2689  0.5314 -1.2894
#>  0.7838 -0.4598  0.2605  0.3141
#>  0.3302 -0.3092 -1.3606 -1.0281
#> -0.1029 -0.4606  0.5504 -0.5991
#>  0.3339 -0.1434  0.0885  0.3569
#> -0.5634  0.4602 -0.6585 -0.5354
#>  0.8441  0.9097 -0.3583  0.6452
#> -1.2043 -0.5838  0.3814  1.4960
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> 0.01 *
#> -9.9543
#>  0.7549
#> -8.3099
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.4152
#>  1.0710
#> -0.8902
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.4422
#>  0.1899
#>  0.1474
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.8595
#> -2.0373
#>  0.2836
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.6798
#> -1.1397
#>  0.8287
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.9204
#>  3.2267
#>  0.5465
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.4931
#> -1.4769
#>  0.4412
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.1975
#>  1.0870
#>  1.4432
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.3773
#> -0.4296
#>  0.2689
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.4434
#> -2.1599
#>  1.4137
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.2703
#> -0.0146
#>  1.4458
#> -0.6430
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.0490
#> -0.8759
#> -0.3446
#>  0.7235
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.2982
#>  0.2689
#>  0.5314
#> -1.2894
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.7838
#> -0.4598
#>  0.2605
#>  0.3141
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.3302
#> -0.3092
#> -1.3606
#> -1.0281
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1029
#> -0.4606
#>  0.5504
#> -0.5991
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.3339
#> -0.1434
#>  0.0885
#>  0.3569
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.5634
#>  0.4602
#> -0.6585
#> -0.5354
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.8441
#>  0.9097
#> -0.3583
#>  0.6452
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.2043
#> -0.5838
#>  0.3814
#>  1.4960
#> [ CPUFloatType{4} ]
#> 
#> 
```
