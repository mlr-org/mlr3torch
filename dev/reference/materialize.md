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
#> -1.6249 -0.9636 -0.1405
#> -0.7969 -1.1135 -0.0671
#>  1.2032  1.5025  2.9158
#>  0.8221  1.5085  0.5806
#> -0.0493  0.3310 -0.9398
#> -0.1803  0.6095  0.9016
#> -0.0885  0.5722 -0.7507
#>  0.4534 -1.1672 -1.4275
#>  0.1996 -0.9000 -0.5968
#> -0.2072  0.1604 -2.0664
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.6249
#> -0.9636
#> -0.1405
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.7969
#> -1.1135
#> -0.0671
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.2032
#>  1.5025
#>  2.9158
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.8221
#>  1.5085
#>  0.5806
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.0493
#>  0.3310
#> -0.9398
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.1803
#>  0.6095
#>  0.9016
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.0885
#>  0.5722
#> -0.7507
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.4534
#> -1.1672
#> -1.4275
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.1996
#> -0.9000
#> -0.5968
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.2072
#>  0.1604
#> -2.0664
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.6249 -0.9636 -0.1405
#> -0.7969 -1.1135 -0.0671
#>  1.2032  1.5025  2.9158
#>  0.8221  1.5085  0.5806
#> -0.0493  0.3310 -0.9398
#> -0.1803  0.6095  0.9016
#> -0.0885  0.5722 -0.7507
#>  0.4534 -1.1672 -1.4275
#>  0.1996 -0.9000 -0.5968
#> -0.2072  0.1604 -2.0664
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.7204 -0.3349  0.8845  1.0884
#>  1.2363 -0.1786 -0.3500  0.9815
#> -1.4250 -0.2998 -1.0120  0.9384
#> -1.5942 -0.6926 -1.0827  0.0801
#> -0.8424  0.7055 -1.0166  0.2515
#>  0.1127 -0.9203 -0.1473 -0.3680
#>  0.1178  1.0480 -0.9867 -0.7865
#>  0.0200  0.4855 -1.2171  1.6226
#> -3.3723 -0.9555  0.4094  0.8759
#> -0.3960  0.6760 -0.1463  0.8920
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.6249
#> -0.9636
#> -0.1405
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.7969
#> -1.1135
#> -0.0671
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.2032
#>  1.5025
#>  2.9158
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.8221
#>  1.5085
#>  0.5806
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.0493
#>  0.3310
#> -0.9398
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.1803
#>  0.6095
#>  0.9016
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.0885
#>  0.5722
#> -0.7507
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.4534
#> -1.1672
#> -1.4275
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.1996
#> -0.9000
#> -0.5968
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.2072
#>  0.1604
#> -2.0664
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.7204
#> -0.3349
#>  0.8845
#>  1.0884
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.2363
#> -0.1786
#> -0.3500
#>  0.9815
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.4250
#> -0.2998
#> -1.0120
#>  0.9384
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.5942
#> -0.6926
#> -1.0827
#>  0.0801
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.8424
#>  0.7055
#> -1.0166
#>  0.2515
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.1127
#> -0.9203
#> -0.1473
#> -0.3680
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.1178
#>  1.0480
#> -0.9867
#> -0.7865
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.0200
#>  0.4855
#> -1.2171
#>  1.6226
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -3.3723
#> -0.9555
#>  0.4094
#>  0.8759
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.3960
#>  0.6760
#> -0.1463
#>  0.8920
#> [ CPUFloatType{4} ]
#> 
#> 
```
