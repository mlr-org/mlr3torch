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
#>  1.1275 -0.4958  0.7321
#>  0.1486  0.1652 -0.3897
#>  0.7350  0.3670 -0.9614
#> -0.2349  0.3877  0.0927
#> -0.4541 -0.8795 -0.2641
#>  0.4433  0.8873 -1.5336
#> -0.3806 -0.6843  0.4105
#> -0.1768 -0.7476  1.4551
#>  0.3209 -0.2974 -0.4998
#>  0.1113  0.6158  0.3974
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.1275
#> -0.4958
#>  0.7321
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.1486
#>  0.1652
#> -0.3897
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.7350
#>  0.3670
#> -0.9614
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.2349
#>  0.3877
#>  0.0927
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.4541
#> -0.8795
#> -0.2641
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.4433
#>  0.8873
#> -1.5336
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.3806
#> -0.6843
#>  0.4105
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.1768
#> -0.7476
#>  1.4551
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.3209
#> -0.2974
#> -0.4998
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.1113
#>  0.6158
#>  0.3974
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.1275 -0.4958  0.7321
#>  0.1486  0.1652 -0.3897
#>  0.7350  0.3670 -0.9614
#> -0.2349  0.3877  0.0927
#> -0.4541 -0.8795 -0.2641
#>  0.4433  0.8873 -1.5336
#> -0.3806 -0.6843  0.4105
#> -0.1768 -0.7476  1.4551
#>  0.3209 -0.2974 -0.4998
#>  0.1113  0.6158  0.3974
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.5952 -1.2967 -0.9117  0.2672
#>  0.4197 -0.2130  0.2615 -2.4433
#> -0.2367  0.9032  0.2431 -0.7365
#>  0.3358 -0.0281  0.5099 -1.9949
#>  1.9405  0.3398 -0.6006  0.1127
#>  0.3211  1.3029  1.4661  0.5819
#>  0.6866 -1.5837  0.3930 -0.2545
#> -0.0612 -1.2920 -0.3756 -1.7262
#> -0.5995 -2.1505  2.2271  0.6283
#> -0.2564 -1.6008 -2.2118  1.9446
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.1275
#> -0.4958
#>  0.7321
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.1486
#>  0.1652
#> -0.3897
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.7350
#>  0.3670
#> -0.9614
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.2349
#>  0.3877
#>  0.0927
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.4541
#> -0.8795
#> -0.2641
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.4433
#>  0.8873
#> -1.5336
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.3806
#> -0.6843
#>  0.4105
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.1768
#> -0.7476
#>  1.4551
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.3209
#> -0.2974
#> -0.4998
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.1113
#>  0.6158
#>  0.3974
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.5952
#> -1.2967
#> -0.9117
#>  0.2672
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.4197
#> -0.2130
#>  0.2615
#> -2.4433
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.2367
#>  0.9032
#>  0.2431
#> -0.7365
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.3358
#> -0.0281
#>  0.5099
#> -1.9949
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.9405
#>  0.3398
#> -0.6006
#>  0.1127
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.3211
#>  1.3029
#>  1.4661
#>  0.5819
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.6866
#> -1.5837
#>  0.3930
#> -0.2545
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> 0.01 *
#> -6.1223
#> -129.2011
#> -37.5606
#> -172.6216
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.5995
#> -2.1505
#>  2.2271
#>  0.6283
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.2564
#> -1.6008
#> -2.2118
#>  1.9446
#> [ CPUFloatType{4} ]
#> 
#> 
```
