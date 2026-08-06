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
#>  1.4947  0.0655  0.3660
#>  1.0479 -0.2152 -0.0659
#> -0.5155 -0.9822 -1.0197
#> -0.9807 -1.6585  0.8409
#> -0.1761  1.8671 -0.5066
#>  1.3455  1.1678 -0.7975
#> -0.1366 -0.9656 -1.6349
#> -1.2250 -0.3723 -0.3602
#>  0.3340 -0.0234  0.7374
#> -1.3524  0.8567 -0.9775
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.4947
#>  0.0655
#>  0.3660
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.0479
#> -0.2152
#> -0.0659
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.5155
#> -0.9822
#> -1.0197
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.9807
#> -1.6585
#>  0.8409
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.1761
#>  1.8671
#> -0.5066
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.3455
#>  1.1678
#> -0.7975
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.1366
#> -0.9656
#> -1.6349
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.2250
#> -0.3723
#> -0.3602
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.3340
#> -0.0234
#>  0.7374
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.3524
#>  0.8567
#> -0.9775
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.4947  0.0655  0.3660
#>  1.0479 -0.2152 -0.0659
#> -0.5155 -0.9822 -1.0197
#> -0.9807 -1.6585  0.8409
#> -0.1761  1.8671 -0.5066
#>  1.3455  1.1678 -0.7975
#> -0.1366 -0.9656 -1.6349
#> -1.2250 -0.3723 -0.3602
#>  0.3340 -0.0234  0.7374
#> -1.3524  0.8567 -0.9775
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.9477 -0.3833 -0.6453 -0.0121
#> -1.3279 -0.4008 -1.5955 -1.6926
#> -0.1897 -0.6985  1.3267  1.1160
#> -0.8336 -1.2070  1.2724  0.9883
#>  0.3899 -0.7924  0.8493  1.5158
#>  0.4767  0.9478 -0.3487 -1.5304
#>  0.9463  1.3177  0.2667 -0.9854
#>  0.8103  0.5597  0.0089 -0.2524
#>  1.0959  0.6218  0.4950  1.6609
#>  1.3662  0.0990 -2.8191 -0.7745
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.4947
#>  0.0655
#>  0.3660
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.0479
#> -0.2152
#> -0.0659
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.5155
#> -0.9822
#> -1.0197
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.9807
#> -1.6585
#>  0.8409
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.1761
#>  1.8671
#> -0.5066
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.3455
#>  1.1678
#> -0.7975
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.1366
#> -0.9656
#> -1.6349
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.2250
#> -0.3723
#> -0.3602
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.3340
#> -0.0234
#>  0.7374
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.3524
#>  0.8567
#> -0.9775
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.9477
#> -0.3833
#> -0.6453
#> -0.0121
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.3279
#> -0.4008
#> -1.5955
#> -1.6926
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.1897
#> -0.6985
#>  1.3267
#>  1.1160
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.8336
#> -1.2070
#>  1.2724
#>  0.9883
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.3899
#> -0.7924
#>  0.8493
#>  1.5158
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4767
#>  0.9478
#> -0.3487
#> -1.5304
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.9463
#>  1.3177
#>  0.2667
#> -0.9854
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.8103
#>  0.5597
#>  0.0089
#> -0.2524
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.0959
#>  0.6218
#>  0.4950
#>  1.6609
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.3662
#>  0.0990
#> -2.8191
#> -0.7745
#> [ CPUFloatType{4} ]
#> 
#> 
```
