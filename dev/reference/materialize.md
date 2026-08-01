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
#> -0.8895  0.2687 -0.4977
#> -0.5897  0.4569  1.0439
#>  1.2595 -0.6420  0.7359
#>  0.3143 -0.6286  0.9630
#> -1.7329 -0.1853  0.7220
#>  0.2544  0.4697  1.2010
#> -0.6299  0.0335  1.7566
#>  1.3892  0.6840  0.7942
#>  0.2207 -1.4126 -1.2628
#>  0.7064  1.3803 -0.3922
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.8895
#>  0.2687
#> -0.4977
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.5897
#>  0.4569
#>  1.0439
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.2595
#> -0.6420
#>  0.7359
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.3143
#> -0.6286
#>  0.9630
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.7329
#> -0.1853
#>  0.7220
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.2544
#>  0.4697
#>  1.2010
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.6299
#>  0.0335
#>  1.7566
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.3892
#>  0.6840
#>  0.7942
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.2207
#> -1.4126
#> -1.2628
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.7064
#>  1.3803
#> -0.3922
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.8895  0.2687 -0.4977
#> -0.5897  0.4569  1.0439
#>  1.2595 -0.6420  0.7359
#>  0.3143 -0.6286  0.9630
#> -1.7329 -0.1853  0.7220
#>  0.2544  0.4697  1.2010
#> -0.6299  0.0335  1.7566
#>  1.3892  0.6840  0.7942
#>  0.2207 -1.4126 -1.2628
#>  0.7064  1.3803 -0.3922
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.1550 -0.8679  0.8856  1.9405
#> -0.6972  0.5798 -1.3768 -1.3351
#>  0.1438  0.9733  0.1894  0.1137
#> -0.3672 -0.6547  0.6374  0.5262
#>  0.3277 -0.8744  1.5814 -0.6431
#> -0.9094  1.0311  0.5896 -0.0552
#> -0.6321 -1.8752 -0.6995 -1.2671
#>  1.5983 -1.6973  1.0936  0.3768
#> -0.7502 -0.5225 -1.6189  0.3041
#> -1.3079  0.0707 -0.4965 -0.7682
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.8895
#>  0.2687
#> -0.4977
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.5897
#>  0.4569
#>  1.0439
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.2595
#> -0.6420
#>  0.7359
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.3143
#> -0.6286
#>  0.9630
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.7329
#> -0.1853
#>  0.7220
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.2544
#>  0.4697
#>  1.2010
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.6299
#>  0.0335
#>  1.7566
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.3892
#>  0.6840
#>  0.7942
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.2207
#> -1.4126
#> -1.2628
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.7064
#>  1.3803
#> -0.3922
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.1550
#> -0.8679
#>  0.8856
#>  1.9405
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.6972
#>  0.5798
#> -1.3768
#> -1.3351
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.1438
#>  0.9733
#>  0.1894
#>  0.1137
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.3672
#> -0.6547
#>  0.6374
#>  0.5262
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.3277
#> -0.8744
#>  1.5814
#> -0.6431
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.9094
#>  1.0311
#>  0.5896
#> -0.0552
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.6321
#> -1.8752
#> -0.6995
#> -1.2671
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.5983
#> -1.6973
#>  1.0936
#>  0.3768
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.7502
#> -0.5225
#> -1.6189
#>  0.3041
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.3079
#>  0.0707
#> -0.4965
#> -0.7682
#> [ CPUFloatType{4} ]
#> 
#> 
```
