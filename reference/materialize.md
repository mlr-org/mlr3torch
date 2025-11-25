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
#>  0.3410 -0.5478  0.4420
#>  0.8284  1.2942 -0.2742
#> -0.8387 -1.5603  0.8534
#> -0.5863 -0.7297  0.2916
#> -0.1709 -1.1913  0.0189
#> -1.3408 -0.0253  1.7381
#> -1.4870 -1.5383 -0.2885
#> -1.0551 -1.0424  0.8519
#> -0.9033  0.0312 -0.5686
#>  0.2981 -1.2688 -1.1190
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3410
#> -0.5478
#>  0.4420
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.8284
#>  1.2942
#> -0.2742
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.8387
#> -1.5603
#>  0.8534
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.5863
#> -0.7297
#>  0.2916
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.1709
#> -1.1913
#>  0.0189
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -1.3408
#> -0.0253
#>  1.7381
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -1.4870
#> -1.5383
#> -0.2885
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.0551
#> -1.0424
#>  0.8519
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.9033
#>  0.0312
#> -0.5686
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.2981
#> -1.2688
#> -1.1190
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3410 -0.5478  0.4420
#>  0.8284  1.2942 -0.2742
#> -0.8387 -1.5603  0.8534
#> -0.5863 -0.7297  0.2916
#> -0.1709 -1.1913  0.0189
#> -1.3408 -0.0253  1.7381
#> -1.4870 -1.5383 -0.2885
#> -1.0551 -1.0424  0.8519
#> -0.9033  0.0312 -0.5686
#>  0.2981 -1.2688 -1.1190
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2324 -1.6235  1.7203 -0.8513
#>  0.4663  1.0089  1.0003 -0.1445
#>  0.9465 -0.0024 -0.5093  0.4063
#>  0.5440 -0.8111 -0.4219 -1.4323
#>  0.5833  0.1734 -0.7890 -0.4627
#> -1.1651 -1.5650 -2.4204 -1.0971
#> -0.7232  0.2837 -0.0969 -0.4386
#>  0.4146 -0.8358  0.2442  1.3394
#>  0.0873  0.7175 -0.2835 -0.1161
#>  0.5496  0.2338 -1.1196  0.9898
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3410
#> -0.5478
#>  0.4420
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.8284
#>  1.2942
#> -0.2742
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.8387
#> -1.5603
#>  0.8534
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.5863
#> -0.7297
#>  0.2916
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.1709
#> -1.1913
#>  0.0189
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -1.3408
#> -0.0253
#>  1.7381
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -1.4870
#> -1.5383
#> -0.2885
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.0551
#> -1.0424
#>  0.8519
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.9033
#>  0.0312
#> -0.5686
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.2981
#> -1.2688
#> -1.1190
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2324
#> -1.6235
#>  1.7203
#> -0.8513
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.4663
#>  1.0089
#>  1.0003
#> -0.1445
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.9465
#> -0.0024
#> -0.5093
#>  0.4063
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.5440
#> -0.8111
#> -0.4219
#> -1.4323
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.5833
#>  0.1734
#> -0.7890
#> -0.4627
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.1651
#> -1.5650
#> -2.4204
#> -1.0971
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.7232
#>  0.2837
#> -0.0969
#> -0.4386
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4146
#> -0.8358
#>  0.2442
#>  1.3394
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.0873
#>  0.7175
#> -0.2835
#> -0.1161
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.5496
#>  0.2338
#> -1.1196
#>  0.9898
#> [ CPUFloatType{4} ]
#> 
#> 
```
