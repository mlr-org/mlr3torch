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
#>  0.9225  2.2327 -0.7039
#> -1.3261 -0.5701  0.5796
#>  0.9495 -1.5190  1.2719
#>  0.5473  0.1603 -0.2017
#> -0.0792  0.8376  0.9449
#> -0.2632  0.0088 -0.7655
#>  0.9407 -0.8682  0.2335
#>  0.7823  1.1253 -0.1434
#>  0.3549 -0.4419  1.9403
#> -0.5474  0.9559 -1.7793
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.9225
#>  2.2327
#> -0.7039
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.3261
#> -0.5701
#>  0.5796
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.9495
#> -1.5190
#>  1.2719
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.5473
#>  0.1603
#> -0.2017
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.0792
#>  0.8376
#>  0.9449
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.2632
#>  0.0088
#> -0.7655
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.9407
#> -0.8682
#>  0.2335
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.7823
#>  1.1253
#> -0.1434
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.3549
#> -0.4419
#>  1.9403
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.5474
#>  0.9559
#> -1.7793
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.9225  2.2327 -0.7039
#> -1.3261 -0.5701  0.5796
#>  0.9495 -1.5190  1.2719
#>  0.5473  0.1603 -0.2017
#> -0.0792  0.8376  0.9449
#> -0.2632  0.0088 -0.7655
#>  0.9407 -0.8682  0.2335
#>  0.7823  1.1253 -0.1434
#>  0.3549 -0.4419  1.9403
#> -0.5474  0.9559 -1.7793
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.1060  0.4986  0.2737  0.5996
#> -1.1947 -2.8579 -1.0485 -0.2744
#> -0.5617  0.5595  0.1483 -0.2581
#>  1.0623 -1.4268 -0.2735 -1.2453
#>  0.3442 -0.0836  0.9986  1.2672
#> -1.2924 -0.9194  0.2143  0.2041
#>  0.7119  0.2178 -0.0935 -3.3770
#>  0.4801 -0.3224  2.1881  0.2396
#>  0.1784  0.6602  0.8551  0.8267
#> -1.5305  0.3625 -0.7255 -1.2492
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.9225
#>  2.2327
#> -0.7039
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.3261
#> -0.5701
#>  0.5796
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.9495
#> -1.5190
#>  1.2719
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.5473
#>  0.1603
#> -0.2017
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.0792
#>  0.8376
#>  0.9449
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.2632
#>  0.0088
#> -0.7655
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.9407
#> -0.8682
#>  0.2335
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.7823
#>  1.1253
#> -0.1434
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.3549
#> -0.4419
#>  1.9403
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.5474
#>  0.9559
#> -1.7793
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.1060
#>  0.4986
#>  0.2737
#>  0.5996
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.1947
#> -2.8579
#> -1.0485
#> -0.2744
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.5617
#>  0.5595
#>  0.1483
#> -0.2581
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.0623
#> -1.4268
#> -0.2735
#> -1.2453
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.3442
#> -0.0836
#>  0.9986
#>  1.2672
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.2924
#> -0.9194
#>  0.2143
#>  0.2041
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.7119
#>  0.2178
#> -0.0935
#> -3.3770
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4801
#> -0.3224
#>  2.1881
#>  0.2396
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.1784
#>  0.6602
#>  0.8551
#>  0.8267
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.5305
#>  0.3625
#> -0.7255
#> -1.2492
#> [ CPUFloatType{4} ]
#> 
#> 
```
