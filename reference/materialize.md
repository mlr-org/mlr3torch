# Materialize Lazy Tensor Columns

This will materialize a
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
or a [`data.frame()`](https://rdrr.io/r/base/data.frame.html) /
[`list()`](https://rdrr.io/r/base/list.html) containing – among other
things –
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
columns. I.e. the data described in the underlying
[`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)s
is loaded for the indices in the
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md),
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
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
  or a [`list()`](https://rdrr.io/r/base/list.html) /
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html) containing
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)s
or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md))

## Details

Materializing a lazy tensor consists of:

1.  Loading the data from the internal dataset of the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md).

2.  Processing these batches in the preprocessing
    [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s.

3.  Returning the result of the
    [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
    pointed to by the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)
    (`pointer`).

With multiple
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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
#> -0.9998  1.3690  0.2783
#>  0.6143  0.1464  0.6510
#> -0.0043 -0.6607 -0.1144
#>  0.2510  0.6051  1.2596
#>  1.3518  0.6979 -0.0623
#>  0.1810 -0.7625 -0.2971
#>  0.3792 -1.0608 -0.2672
#>  0.2921  1.2106 -1.2185
#> -0.7106 -1.1398 -1.0669
#> -1.4973  0.2942  0.6314
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.9998
#>  1.3690
#>  0.2783
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.6143
#>  0.1464
#>  0.6510
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> 0.001 *
#> -4.2864
#> -660.6835
#> -114.3686
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.2510
#>  0.6051
#>  1.2596
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.3518
#>  0.6979
#> -0.0623
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.1810
#> -0.7625
#> -0.2971
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.3792
#> -1.0608
#> -0.2672
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.2921
#>  1.2106
#> -1.2185
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.7106
#> -1.1398
#> -1.0669
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.4973
#>  0.2942
#>  0.6314
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.9998  1.3690  0.2783
#>  0.6143  0.1464  0.6510
#> -0.0043 -0.6607 -0.1144
#>  0.2510  0.6051  1.2596
#>  1.3518  0.6979 -0.0623
#>  0.1810 -0.7625 -0.2971
#>  0.3792 -1.0608 -0.2672
#>  0.2921  1.2106 -1.2185
#> -0.7106 -1.1398 -1.0669
#> -1.4973  0.2942  0.6314
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0948  1.6054 -1.6551 -0.6419
#>  0.8432  0.4529  0.5160 -0.5908
#>  1.3262  0.7090  0.1610  0.5235
#> -1.0406  1.7880  1.5747 -0.5748
#>  2.2233 -0.5783  0.7422  0.1636
#>  1.4031 -1.2452 -1.1952 -1.1571
#>  0.1680  0.3079  0.6436  1.5532
#>  0.6803  0.7584 -1.2660 -1.0233
#> -0.3231  0.9361 -1.4921  0.2539
#> -0.3917  0.7497 -0.2370  1.5893
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.9998
#>  1.3690
#>  0.2783
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.6143
#>  0.1464
#>  0.6510
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> 0.001 *
#> -4.2864
#> -660.6835
#> -114.3686
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.2510
#>  0.6051
#>  1.2596
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.3518
#>  0.6979
#> -0.0623
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.1810
#> -0.7625
#> -0.2971
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.3792
#> -1.0608
#> -0.2672
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.2921
#>  1.2106
#> -1.2185
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.7106
#> -1.1398
#> -1.0669
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.4973
#>  0.2942
#>  0.6314
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0948
#>  1.6054
#> -1.6551
#> -0.6419
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.8432
#>  0.4529
#>  0.5160
#> -0.5908
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.3262
#>  0.7090
#>  0.1610
#>  0.5235
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.0406
#>  1.7880
#>  1.5747
#> -0.5748
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  2.2233
#> -0.5783
#>  0.7422
#>  0.1636
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.4031
#> -1.2452
#> -1.1952
#> -1.1571
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.1680
#>  0.3079
#>  0.6436
#>  1.5532
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.6803
#>  0.7584
#> -1.2660
#> -1.0233
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.3231
#>  0.9361
#> -1.4921
#>  0.2539
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.3917
#>  0.7497
#> -0.2370
#>  1.5893
#> [ CPUFloatType{4} ]
#> 
#> 
```
