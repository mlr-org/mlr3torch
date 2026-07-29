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
#> -0.5783  1.5648  0.8126
#> -0.1391  0.6230  1.7984
#> -1.3604  0.5193  0.1954
#>  0.0804 -0.5199  0.7960
#> -1.3186 -0.4287 -1.5513
#>  0.0328 -0.4144 -1.6015
#>  2.6577  0.9005 -1.6721
#>  0.7454 -0.3849  0.0257
#>  0.0787 -0.1653 -0.2070
#>  0.2631  0.0862 -1.8661
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.5783
#>  1.5648
#>  0.8126
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.1391
#>  0.6230
#>  1.7984
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.3604
#>  0.5193
#>  0.1954
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.0804
#> -0.5199
#>  0.7960
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.3186
#> -0.4287
#> -1.5513
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.0328
#> -0.4144
#> -1.6015
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  2.6577
#>  0.9005
#> -1.6721
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.7454
#> -0.3849
#>  0.0257
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.0787
#> -0.1653
#> -0.2070
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.2631
#>  0.0862
#> -1.8661
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.5783  1.5648  0.8126
#> -0.1391  0.6230  1.7984
#> -1.3604  0.5193  0.1954
#>  0.0804 -0.5199  0.7960
#> -1.3186 -0.4287 -1.5513
#>  0.0328 -0.4144 -1.6015
#>  2.6577  0.9005 -1.6721
#>  0.7454 -0.3849  0.0257
#>  0.0787 -0.1653 -0.2070
#>  0.2631  0.0862 -1.8661
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.1552  0.5467  0.2360  0.8581
#> -0.0173  0.0955 -0.0185  0.8094
#>  0.4221  1.3905  0.1301 -1.6531
#> -1.4489  0.7493  0.7999 -1.0202
#>  1.9948  0.6520  0.2519  2.0920
#>  0.7183  0.3076  1.0248  0.2850
#>  0.7783 -0.5387 -1.6133 -0.0080
#> -0.4251 -1.1394  1.1041 -0.5484
#>  2.2339 -0.1429 -1.3458 -1.6657
#>  0.3426  0.1062  0.2729  1.0155
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.5783
#>  1.5648
#>  0.8126
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.1391
#>  0.6230
#>  1.7984
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.3604
#>  0.5193
#>  0.1954
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.0804
#> -0.5199
#>  0.7960
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.3186
#> -0.4287
#> -1.5513
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.0328
#> -0.4144
#> -1.6015
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  2.6577
#>  0.9005
#> -1.6721
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.7454
#> -0.3849
#>  0.0257
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.0787
#> -0.1653
#> -0.2070
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.2631
#>  0.0862
#> -1.8661
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.1552
#>  0.5467
#>  0.2360
#>  0.8581
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.0173
#>  0.0955
#> -0.0185
#>  0.8094
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.4221
#>  1.3905
#>  0.1301
#> -1.6531
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.4489
#>  0.7493
#>  0.7999
#> -1.0202
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.9948
#>  0.6520
#>  0.2519
#>  2.0920
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.7183
#>  0.3076
#>  1.0248
#>  0.2850
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.7783
#> -0.5387
#> -1.6133
#> -0.0080
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.4251
#> -1.1394
#>  1.1041
#> -0.5484
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  2.2339
#> -0.1429
#> -1.3458
#> -1.6657
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.3426
#>  0.1062
#>  0.2729
#>  1.0155
#> [ CPUFloatType{4} ]
#> 
#> 
```
