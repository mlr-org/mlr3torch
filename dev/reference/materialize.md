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
#> -0.6432 -0.1911 -0.2109
#> -0.9750 -1.0694  0.3755
#>  1.8703 -0.2001  0.4859
#>  0.7598  0.0038 -1.3262
#> -1.1244  0.2343  1.2003
#> -0.3233 -0.1043 -1.6293
#> -0.3199  1.2685  0.8751
#> -0.8008 -1.4030  1.2216
#> -1.5355  0.0376 -0.0468
#>  0.4954 -0.6278  0.9463
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.6432
#> -0.1911
#> -0.2109
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.9750
#> -1.0694
#>  0.3755
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.8703
#> -0.2001
#>  0.4859
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7598
#>  0.0038
#> -1.3262
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.1244
#>  0.2343
#>  1.2003
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.3233
#> -0.1043
#> -1.6293
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.3199
#>  1.2685
#>  0.8751
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.8008
#> -1.4030
#>  1.2216
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.5355
#>  0.0376
#> -0.0468
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4954
#> -0.6278
#>  0.9463
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.6432 -0.1911 -0.2109
#> -0.9750 -1.0694  0.3755
#>  1.8703 -0.2001  0.4859
#>  0.7598  0.0038 -1.3262
#> -1.1244  0.2343  1.2003
#> -0.3233 -0.1043 -1.6293
#> -0.3199  1.2685  0.8751
#> -0.8008 -1.4030  1.2216
#> -1.5355  0.0376 -0.0468
#>  0.4954 -0.6278  0.9463
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.6663  0.5538  0.9882  0.5300
#> -0.6698  0.8444  0.4511 -0.5611
#> -0.8856 -0.0631  0.0028  0.1085
#> -1.2518  0.4391 -2.3730  1.7491
#>  2.2451 -1.5264 -0.2087 -1.3965
#>  0.5322  0.9783  0.2675  1.5290
#> -1.4771  1.8939 -1.0890  0.0985
#>  0.5891  0.3992 -0.2343  1.0288
#> -1.0181  1.1361 -0.8677  0.3452
#>  0.1844 -0.3377 -0.6686 -1.0793
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.6432
#> -0.1911
#> -0.2109
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.9750
#> -1.0694
#>  0.3755
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.8703
#> -0.2001
#>  0.4859
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7598
#>  0.0038
#> -1.3262
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.1244
#>  0.2343
#>  1.2003
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.3233
#> -0.1043
#> -1.6293
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.3199
#>  1.2685
#>  0.8751
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.8008
#> -1.4030
#>  1.2216
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.5355
#>  0.0376
#> -0.0468
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4954
#> -0.6278
#>  0.9463
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.6663
#>  0.5538
#>  0.9882
#>  0.5300
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.6698
#>  0.8444
#>  0.4511
#> -0.5611
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.8856
#> -0.0631
#>  0.0028
#>  0.1085
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.2518
#>  0.4391
#> -2.3730
#>  1.7491
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  2.2451
#> -1.5264
#> -0.2087
#> -1.3965
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.5322
#>  0.9783
#>  0.2675
#>  1.5290
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.4771
#>  1.8939
#> -1.0890
#>  0.0985
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.5891
#>  0.3992
#> -0.2343
#>  1.0288
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.0181
#>  1.1361
#> -0.8677
#>  0.3452
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.1844
#> -0.3377
#> -0.6686
#> -1.0793
#> [ CPUFloatType{4} ]
#> 
#> 
```
