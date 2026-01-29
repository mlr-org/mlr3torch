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
#>  0.0548 -2.0001  0.1621
#> -1.8471  0.0306  1.4816
#> -0.0555  0.6110 -0.3156
#> -0.6056  0.3216  0.6922
#>  1.5041 -1.0019  1.1376
#> -0.4767 -0.5134 -1.2800
#> -0.4358  0.2021  0.0617
#>  2.0289 -0.1365 -0.1119
#> -0.4336  0.5342  1.2269
#> -0.7808  1.0000  0.0739
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.0548
#> -2.0001
#>  0.1621
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.8471
#>  0.0306
#>  1.4816
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.0555
#>  0.6110
#> -0.3156
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6056
#>  0.3216
#>  0.6922
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.5041
#> -1.0019
#>  1.1376
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.4767
#> -0.5134
#> -1.2800
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.4358
#>  0.2021
#>  0.0617
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  2.0289
#> -0.1365
#> -0.1119
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.4336
#>  0.5342
#>  1.2269
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.7808
#>  1.0000
#>  0.0739
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.0548 -2.0001  0.1621
#> -1.8471  0.0306  1.4816
#> -0.0555  0.6110 -0.3156
#> -0.6056  0.3216  0.6922
#>  1.5041 -1.0019  1.1376
#> -0.4767 -0.5134 -1.2800
#> -0.4358  0.2021  0.0617
#>  2.0289 -0.1365 -0.1119
#> -0.4336  0.5342  1.2269
#> -0.7808  1.0000  0.0739
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.3160  0.6118 -0.9717  0.1837
#> -0.1518  0.1891  0.3759 -1.3565
#>  0.5991 -1.2876  0.9732 -1.5456
#> -0.4114  0.1574 -0.7178  0.0739
#>  0.3309 -0.3937  0.3191  1.5122
#>  0.2168 -0.2756 -1.4510  0.0937
#>  2.0964  0.6350  1.1514  1.5170
#> -0.1564  1.5366  1.1535  0.6629
#> -0.2502 -0.6394  0.0294  0.5373
#> -0.0096  0.6451 -0.5569  0.2409
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.0548
#> -2.0001
#>  0.1621
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.8471
#>  0.0306
#>  1.4816
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.0555
#>  0.6110
#> -0.3156
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6056
#>  0.3216
#>  0.6922
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.5041
#> -1.0019
#>  1.1376
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.4767
#> -0.5134
#> -1.2800
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.4358
#>  0.2021
#>  0.0617
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  2.0289
#> -0.1365
#> -0.1119
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.4336
#>  0.5342
#>  1.2269
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.7808
#>  1.0000
#>  0.0739
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.3160
#>  0.6118
#> -0.9717
#>  0.1837
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.1518
#>  0.1891
#>  0.3759
#> -1.3565
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5991
#> -1.2876
#>  0.9732
#> -1.5456
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.4114
#>  0.1574
#> -0.7178
#>  0.0739
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.3309
#> -0.3937
#>  0.3191
#>  1.5122
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.2168
#> -0.2756
#> -1.4510
#>  0.0937
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  2.0964
#>  0.6350
#>  1.1514
#>  1.5170
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.1564
#>  1.5366
#>  1.1535
#>  0.6629
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.2502
#> -0.6394
#>  0.0294
#>  0.5373
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.0096
#>  0.6451
#> -0.5569
#>  0.2409
#> [ CPUFloatType{4} ]
#> 
#> 
```
