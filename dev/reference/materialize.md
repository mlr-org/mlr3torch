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
#> -1.1160 -0.9015 -1.1527
#>  0.4315  0.2257 -0.6495
#>  0.4114  0.1612 -1.1283
#>  0.7368  0.5841  0.5345
#> -0.8858  0.1973 -0.2821
#> -0.3706 -0.4344 -1.9972
#>  0.4535 -0.8490 -0.4003
#>  0.4356  2.4676 -0.8674
#> -1.5587  1.7387 -0.4312
#> -0.3515  1.0644  0.6967
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.1160
#> -0.9015
#> -1.1527
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.4315
#>  0.2257
#> -0.6495
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.4114
#>  0.1612
#> -1.1283
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7368
#>  0.5841
#>  0.5345
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.8858
#>  0.1973
#> -0.2821
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.3706
#> -0.4344
#> -1.9972
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.4535
#> -0.8490
#> -0.4003
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.4356
#>  2.4676
#> -0.8674
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.5587
#>  1.7387
#> -0.4312
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.3515
#>  1.0644
#>  0.6967
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.1160 -0.9015 -1.1527
#>  0.4315  0.2257 -0.6495
#>  0.4114  0.1612 -1.1283
#>  0.7368  0.5841  0.5345
#> -0.8858  0.1973 -0.2821
#> -0.3706 -0.4344 -1.9972
#>  0.4535 -0.8490 -0.4003
#>  0.4356  2.4676 -0.8674
#> -1.5587  1.7387 -0.4312
#> -0.3515  1.0644  0.6967
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0895 -1.3725  0.7236  2.7246
#>  0.7235 -0.2678  0.4570  1.1227
#> -0.8535  0.0427 -0.1781  0.6219
#>  0.6758 -0.1784  1.5907 -0.4406
#> -0.6595  0.6553  1.4374 -0.2368
#>  0.3168 -0.7197 -0.3020  1.5783
#> -0.9951  1.5707  0.0838  0.2606
#>  0.6793 -2.2520  0.2780  0.8630
#> -1.7279 -0.3791 -0.1262 -1.0746
#>  0.5092  0.2877 -0.6658  2.1471
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.1160
#> -0.9015
#> -1.1527
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.4315
#>  0.2257
#> -0.6495
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.4114
#>  0.1612
#> -1.1283
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7368
#>  0.5841
#>  0.5345
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.8858
#>  0.1973
#> -0.2821
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.3706
#> -0.4344
#> -1.9972
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.4535
#> -0.8490
#> -0.4003
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.4356
#>  2.4676
#> -0.8674
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.5587
#>  1.7387
#> -0.4312
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.3515
#>  1.0644
#>  0.6967
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0895
#> -1.3725
#>  0.7236
#>  2.7246
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.7235
#> -0.2678
#>  0.4570
#>  1.1227
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.8535
#>  0.0427
#> -0.1781
#>  0.6219
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.6758
#> -0.1784
#>  1.5907
#> -0.4406
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.6595
#>  0.6553
#>  1.4374
#> -0.2368
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.3168
#> -0.7197
#> -0.3020
#>  1.5783
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.9951
#>  1.5707
#>  0.0838
#>  0.2606
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.6793
#> -2.2520
#>  0.2780
#>  0.8630
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.7279
#> -0.3791
#> -0.1262
#> -1.0746
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.5092
#>  0.2877
#> -0.6658
#>  2.1471
#> [ CPUFloatType{4} ]
#> 
#> 
```
