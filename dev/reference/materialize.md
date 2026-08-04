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
#> -0.8364 -1.6244  0.1575
#> -1.3794  1.0546 -0.2099
#>  0.3706  2.1043 -0.6304
#> -0.6979 -1.1213  0.8001
#>  0.2922 -0.2434 -0.3870
#>  0.2101  1.2463 -1.3275
#> -0.2622 -0.0438  0.0005
#> -0.0312  0.7509 -2.2942
#>  1.7945 -1.8642  1.3558
#> -0.6566 -0.5112  0.8123
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.8364
#> -1.6244
#>  0.1575
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.3794
#>  1.0546
#> -0.2099
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.3706
#>  2.1043
#> -0.6304
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6979
#> -1.1213
#>  0.8001
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.2922
#> -0.2434
#> -0.3870
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.2101
#>  1.2463
#> -1.3275
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.2622
#> -0.0438
#>  0.0005
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.0312
#>  0.7509
#> -2.2942
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.7945
#> -1.8642
#>  1.3558
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.6566
#> -0.5112
#>  0.8123
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.8364 -1.6244  0.1575
#> -1.3794  1.0546 -0.2099
#>  0.3706  2.1043 -0.6304
#> -0.6979 -1.1213  0.8001
#>  0.2922 -0.2434 -0.3870
#>  0.2101  1.2463 -1.3275
#> -0.2622 -0.0438  0.0005
#> -0.0312  0.7509 -2.2942
#>  1.7945 -1.8642  1.3558
#> -0.6566 -0.5112  0.8123
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0008 -1.4034 -0.0145 -0.9016
#>  1.0219 -0.1601 -0.9041  2.9882
#>  0.5643  0.2752  0.2367 -0.0685
#>  0.1488  1.1662  1.8261  1.0325
#>  1.0751  1.5295 -1.1799 -0.3914
#> -0.2027 -1.2744  1.7354  0.4693
#> -1.2152 -0.3897 -0.3348  2.0404
#> -0.1038 -0.3280 -0.2596 -1.0370
#>  0.3377  0.8877 -1.7651  2.2554
#> -0.3007  1.6520 -1.0836 -0.2949
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.8364
#> -1.6244
#>  0.1575
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.3794
#>  1.0546
#> -0.2099
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.3706
#>  2.1043
#> -0.6304
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6979
#> -1.1213
#>  0.8001
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.2922
#> -0.2434
#> -0.3870
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.2101
#>  1.2463
#> -1.3275
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.2622
#> -0.0438
#>  0.0005
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.0312
#>  0.7509
#> -2.2942
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.7945
#> -1.8642
#>  1.3558
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.6566
#> -0.5112
#>  0.8123
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0008
#> -1.4034
#> -0.0145
#> -0.9016
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.0219
#> -0.1601
#> -0.9041
#>  2.9882
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5643
#>  0.2752
#>  0.2367
#> -0.0685
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.1488
#>  1.1662
#>  1.8261
#>  1.0325
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.0751
#>  1.5295
#> -1.1799
#> -0.3914
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.2027
#> -1.2744
#>  1.7354
#>  0.4693
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.2152
#> -0.3897
#> -0.3348
#>  2.0404
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.1038
#> -0.3280
#> -0.2596
#> -1.0370
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.3377
#>  0.8877
#> -1.7651
#>  2.2554
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.3007
#>  1.6520
#> -1.0836
#> -0.2949
#> [ CPUFloatType{4} ]
#> 
#> 
```
