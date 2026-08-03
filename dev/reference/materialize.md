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
#> -0.0775  1.6639 -0.3528
#> -1.3132  1.9056  0.1119
#>  1.6903  0.5058  1.5401
#> -0.0171 -0.8881 -0.8779
#> -2.2233  0.5801 -0.0508
#>  0.5135 -0.0219 -0.6058
#>  0.8008 -0.4239 -0.1000
#>  0.1650 -0.8208  0.3440
#>  0.6726  1.8856 -0.8515
#>  0.3726 -0.0494  0.2704
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.0775
#>  1.6639
#> -0.3528
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.3132
#>  1.9056
#>  0.1119
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.6903
#>  0.5058
#>  1.5401
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.0171
#> -0.8881
#> -0.8779
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -2.2233
#>  0.5801
#> -0.0508
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.5135
#> -0.0219
#> -0.6058
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.8008
#> -0.4239
#> -0.1000
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.1650
#> -0.8208
#>  0.3440
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.6726
#>  1.8856
#> -0.8515
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.3726
#> -0.0494
#>  0.2704
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.0775  1.6639 -0.3528
#> -1.3132  1.9056  0.1119
#>  1.6903  0.5058  1.5401
#> -0.0171 -0.8881 -0.8779
#> -2.2233  0.5801 -0.0508
#>  0.5135 -0.0219 -0.6058
#>  0.8008 -0.4239 -0.1000
#>  0.1650 -0.8208  0.3440
#>  0.6726  1.8856 -0.8515
#>  0.3726 -0.0494  0.2704
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  2.1465  1.8216  0.0303  1.6763
#> -0.2913  1.7753 -1.6633 -1.1460
#> -1.1192  0.2607 -0.7271  0.8971
#>  0.5721 -0.1734  0.1642  0.2962
#> -1.8483  0.1319  0.7577  0.6902
#> -0.5622 -1.8974 -1.1354 -0.4089
#> -1.6868 -0.5402 -2.0243  0.4747
#> -0.3158  0.4779  0.2478  0.5141
#>  1.5708  0.0700  0.4912  0.2501
#> -0.0193 -0.3215  0.1728 -1.9439
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.0775
#>  1.6639
#> -0.3528
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.3132
#>  1.9056
#>  0.1119
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.6903
#>  0.5058
#>  1.5401
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.0171
#> -0.8881
#> -0.8779
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -2.2233
#>  0.5801
#> -0.0508
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.5135
#> -0.0219
#> -0.6058
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.8008
#> -0.4239
#> -0.1000
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.1650
#> -0.8208
#>  0.3440
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.6726
#>  1.8856
#> -0.8515
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.3726
#> -0.0494
#>  0.2704
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  2.1465
#>  1.8216
#>  0.0303
#>  1.6763
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.2913
#>  1.7753
#> -1.6633
#> -1.1460
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.1192
#>  0.2607
#> -0.7271
#>  0.8971
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.5721
#> -0.1734
#>  0.1642
#>  0.2962
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.8483
#>  0.1319
#>  0.7577
#>  0.6902
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.5622
#> -1.8974
#> -1.1354
#> -0.4089
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.6868
#> -0.5402
#> -2.0243
#>  0.4747
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.3158
#>  0.4779
#>  0.2478
#>  0.5141
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.5708
#>  0.0700
#>  0.4912
#>  0.2501
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.0193
#> -0.3215
#>  0.1728
#> -1.9439
#> [ CPUFloatType{4} ]
#> 
#> 
```
