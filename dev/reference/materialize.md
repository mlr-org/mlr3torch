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
#> -0.2884  1.4583 -0.5739
#>  0.4655  1.2196 -0.7115
#>  1.0174 -0.3732  2.9590
#> -1.6453 -0.0395 -1.2780
#> -1.6316 -0.7763  0.2085
#>  0.5866  2.9423 -1.7859
#> -1.6452  1.2368  0.6513
#> -1.1509  0.3838  0.6269
#>  0.7716 -1.7647 -0.1715
#>  1.3475  0.6836 -0.0876
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.2884
#>  1.4583
#> -0.5739
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.4655
#>  1.2196
#> -0.7115
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.0174
#> -0.3732
#>  2.9590
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.6453
#> -0.0395
#> -1.2780
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.6316
#> -0.7763
#>  0.2085
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.5866
#>  2.9423
#> -1.7859
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -1.6452
#>  1.2368
#>  0.6513
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.1509
#>  0.3838
#>  0.6269
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.7716
#> -1.7647
#> -0.1715
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.3475
#>  0.6836
#> -0.0876
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.2884  1.4583 -0.5739
#>  0.4655  1.2196 -0.7115
#>  1.0174 -0.3732  2.9590
#> -1.6453 -0.0395 -1.2780
#> -1.6316 -0.7763  0.2085
#>  0.5866  2.9423 -1.7859
#> -1.6452  1.2368  0.6513
#> -1.1509  0.3838  0.6269
#>  0.7716 -1.7647 -0.1715
#>  1.3475  0.6836 -0.0876
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.1462 -0.8320  1.4857  1.8188
#>  0.5360  0.7247  0.9910  0.1553
#>  0.5646  0.6100  0.4545 -1.6463
#>  0.2833 -0.4842  0.0224  1.4003
#>  0.0720  0.3649  1.0205  0.3556
#> -0.0646  1.1972 -0.7411  0.7714
#> -0.2768 -1.1173  0.2683 -0.0621
#>  1.5324 -0.4241 -1.8800  1.0467
#>  0.4178 -0.5309 -0.2635  1.5074
#> -0.2301  1.2184 -0.9815 -0.1578
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.2884
#>  1.4583
#> -0.5739
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.4655
#>  1.2196
#> -0.7115
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.0174
#> -0.3732
#>  2.9590
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.6453
#> -0.0395
#> -1.2780
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.6316
#> -0.7763
#>  0.2085
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.5866
#>  2.9423
#> -1.7859
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -1.6452
#>  1.2368
#>  0.6513
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.1509
#>  0.3838
#>  0.6269
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.7716
#> -1.7647
#> -0.1715
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.3475
#>  0.6836
#> -0.0876
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.1462
#> -0.8320
#>  1.4857
#>  1.8188
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.5360
#>  0.7247
#>  0.9910
#>  0.1553
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5646
#>  0.6100
#>  0.4545
#> -1.6463
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2833
#> -0.4842
#>  0.0224
#>  1.4003
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.0720
#>  0.3649
#>  1.0205
#>  0.3556
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.0646
#>  1.1972
#> -0.7411
#>  0.7714
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.2768
#> -1.1173
#>  0.2683
#> -0.0621
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.5324
#> -0.4241
#> -1.8800
#>  1.0467
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.4178
#> -0.5309
#> -0.2635
#>  1.5074
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.2301
#>  1.2184
#> -0.9815
#> -0.1578
#> [ CPUFloatType{4} ]
#> 
#> 
```
