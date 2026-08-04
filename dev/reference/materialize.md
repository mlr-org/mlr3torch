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
#>  0.3941 -0.6305  0.4688
#>  0.1794 -0.5947 -1.1798
#> -0.0290 -0.7036  0.5121
#>  0.8063 -0.2922  1.7156
#>  0.4883  1.8916 -0.7297
#> -0.7460  1.3295 -0.3857
#> -0.4378  0.9586  0.9597
#>  1.0309 -0.8250 -0.3980
#> -0.5629 -0.0087  0.5538
#> -1.2009 -0.0038 -1.6786
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3941
#> -0.6305
#>  0.4688
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.1794
#> -0.5947
#> -1.1798
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.0290
#> -0.7036
#>  0.5121
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.8063
#> -0.2922
#>  1.7156
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.4883
#>  1.8916
#> -0.7297
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.7460
#>  1.3295
#> -0.3857
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.4378
#>  0.9586
#>  0.9597
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.0309
#> -0.8250
#> -0.3980
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.5629
#> -0.0087
#>  0.5538
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.2009
#> -0.0038
#> -1.6786
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3941 -0.6305  0.4688
#>  0.1794 -0.5947 -1.1798
#> -0.0290 -0.7036  0.5121
#>  0.8063 -0.2922  1.7156
#>  0.4883  1.8916 -0.7297
#> -0.7460  1.3295 -0.3857
#> -0.4378  0.9586  0.9597
#>  1.0309 -0.8250 -0.3980
#> -0.5629 -0.0087  0.5538
#> -1.2009 -0.0038 -1.6786
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.3603  2.0945  1.2317 -0.5845
#>  0.2571  0.4517  0.9577 -0.1647
#>  0.4646  0.6475  1.7360  1.3213
#> -0.5182  0.3958  0.4209 -0.8776
#> -0.5709  0.2260  0.8047 -1.2636
#> -0.2706 -0.4812 -0.4031  1.7213
#>  0.5221 -0.1702  0.0865  1.1838
#>  0.3201  1.5116  0.6878  0.8906
#> -0.7485  0.8578  0.2955 -0.3568
#>  0.2391  0.5766  0.0606 -1.4716
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3941
#> -0.6305
#>  0.4688
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.1794
#> -0.5947
#> -1.1798
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.0290
#> -0.7036
#>  0.5121
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.8063
#> -0.2922
#>  1.7156
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.4883
#>  1.8916
#> -0.7297
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.7460
#>  1.3295
#> -0.3857
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.4378
#>  0.9586
#>  0.9597
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.0309
#> -0.8250
#> -0.3980
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.5629
#> -0.0087
#>  0.5538
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.2009
#> -0.0038
#> -1.6786
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.3603
#>  2.0945
#>  1.2317
#> -0.5845
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.2571
#>  0.4517
#>  0.9577
#> -0.1647
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.4646
#>  0.6475
#>  1.7360
#>  1.3213
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.5182
#>  0.3958
#>  0.4209
#> -0.8776
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.5709
#>  0.2260
#>  0.8047
#> -1.2636
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.2706
#> -0.4812
#> -0.4031
#>  1.7213
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.5221
#> -0.1702
#>  0.0865
#>  1.1838
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.3201
#>  1.5116
#>  0.6878
#>  0.8906
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.7485
#>  0.8578
#>  0.2955
#> -0.3568
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.2391
#>  0.5766
#>  0.0606
#> -1.4716
#> [ CPUFloatType{4} ]
#> 
#> 
```
