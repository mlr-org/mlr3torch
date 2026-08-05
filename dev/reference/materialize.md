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
#>  0.4194  0.7157  0.2587
#> -0.5546  0.7364  1.3863
#>  0.5115  1.1488 -0.7388
#> -0.4666 -0.7828  1.5931
#> -0.5576 -1.4915  1.5163
#>  1.9161  0.4279  0.1461
#>  0.0427 -0.2972  2.4999
#> -0.1559 -0.7483  1.4477
#>  0.1734 -1.1349 -1.8890
#> -1.1698  1.3294 -1.6079
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.4194
#>  0.7157
#>  0.2587
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.5546
#>  0.7364
#>  1.3863
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.5115
#>  1.1488
#> -0.7388
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.4666
#> -0.7828
#>  1.5931
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.5576
#> -1.4915
#>  1.5163
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.9161
#>  0.4279
#>  0.1461
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.0427
#> -0.2972
#>  2.4999
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.1559
#> -0.7483
#>  1.4477
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.1734
#> -1.1349
#> -1.8890
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.1698
#>  1.3294
#> -1.6079
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.4194  0.7157  0.2587
#> -0.5546  0.7364  1.3863
#>  0.5115  1.1488 -0.7388
#> -0.4666 -0.7828  1.5931
#> -0.5576 -1.4915  1.5163
#>  1.9161  0.4279  0.1461
#>  0.0427 -0.2972  2.4999
#> -0.1559 -0.7483  1.4477
#>  0.1734 -1.1349 -1.8890
#> -1.1698  1.3294 -1.6079
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.5113 -0.4645  0.3954 -0.3922
#> -1.1903  0.7126 -2.0780 -0.2406
#> -0.0543 -0.4436 -0.0768 -0.3268
#> -0.7373  0.9214  0.6964 -0.4207
#>  0.8741  0.6864  1.7783  0.3098
#> -0.0319 -0.0391 -1.0942 -1.0878
#> -0.7754  0.2978  0.3528 -0.4074
#>  0.3007 -0.5838 -1.2098  0.2076
#>  1.0699  0.6684  0.2258  1.4140
#> -0.6406 -0.0224  1.0919  0.7996
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.4194
#>  0.7157
#>  0.2587
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.5546
#>  0.7364
#>  1.3863
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.5115
#>  1.1488
#> -0.7388
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.4666
#> -0.7828
#>  1.5931
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.5576
#> -1.4915
#>  1.5163
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.9161
#>  0.4279
#>  0.1461
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.0427
#> -0.2972
#>  2.4999
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.1559
#> -0.7483
#>  1.4477
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.1734
#> -1.1349
#> -1.8890
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.1698
#>  1.3294
#> -1.6079
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.5113
#> -0.4645
#>  0.3954
#> -0.3922
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.1903
#>  0.7126
#> -2.0780
#> -0.2406
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.0543
#> -0.4436
#> -0.0768
#> -0.3268
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.7373
#>  0.9214
#>  0.6964
#> -0.4207
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.8741
#>  0.6864
#>  1.7783
#>  0.3098
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.0319
#> -0.0391
#> -1.0942
#> -1.0878
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.7754
#>  0.2978
#>  0.3528
#> -0.4074
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.3007
#> -0.5838
#> -1.2098
#>  0.2076
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.0699
#>  0.6684
#>  0.2258
#>  1.4140
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.6406
#> -0.0224
#>  1.0919
#>  0.7996
#> [ CPUFloatType{4} ]
#> 
#> 
```
