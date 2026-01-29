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
#> -0.5304  0.2023 -1.7790
#> -1.4770  0.6345 -0.0653
#> -0.2969  1.1130 -0.6931
#>  0.3642 -0.3132  1.0254
#> -1.0153 -0.4916 -1.4693
#> -0.7949  1.2432 -1.7355
#>  1.5910  0.5773 -0.5546
#>  1.1313 -0.5313  1.8227
#>  0.8585  0.0966 -1.1285
#> -0.7664 -0.0054  0.3108
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.5304
#>  0.2023
#> -1.7790
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.4770
#>  0.6345
#> -0.0653
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.2969
#>  1.1130
#> -0.6931
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.3642
#> -0.3132
#>  1.0254
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.0153
#> -0.4916
#> -1.4693
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.7949
#>  1.2432
#> -1.7355
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.5910
#>  0.5773
#> -0.5546
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.1313
#> -0.5313
#>  1.8227
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.8585
#>  0.0966
#> -1.1285
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.7664
#> -0.0054
#>  0.3108
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.5304  0.2023 -1.7790
#> -1.4770  0.6345 -0.0653
#> -0.2969  1.1130 -0.6931
#>  0.3642 -0.3132  1.0254
#> -1.0153 -0.4916 -1.4693
#> -0.7949  1.2432 -1.7355
#>  1.5910  0.5773 -0.5546
#>  1.1313 -0.5313  1.8227
#>  0.8585  0.0966 -1.1285
#> -0.7664 -0.0054  0.3108
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.9956  0.0815 -1.3069  0.2880
#> -0.8399 -1.6011 -0.5090 -2.2459
#> -0.2411  0.5102 -0.0886 -0.9739
#> -1.3864 -0.3860  0.3528 -0.0865
#>  0.3530 -1.4395 -2.5759 -1.9576
#>  0.4639 -0.8822  2.7892  0.5885
#> -0.7044  0.4520 -0.5958 -1.5558
#> -1.4735 -0.2830 -1.7554  0.8171
#>  0.7891  0.6901 -1.3893  0.0889
#> -1.7774  0.4797 -1.0913 -0.5842
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.5304
#>  0.2023
#> -1.7790
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.4770
#>  0.6345
#> -0.0653
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.2969
#>  1.1130
#> -0.6931
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.3642
#> -0.3132
#>  1.0254
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.0153
#> -0.4916
#> -1.4693
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.7949
#>  1.2432
#> -1.7355
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.5910
#>  0.5773
#> -0.5546
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.1313
#> -0.5313
#>  1.8227
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.8585
#>  0.0966
#> -1.1285
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.7664
#> -0.0054
#>  0.3108
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.9956
#>  0.0815
#> -1.3069
#>  0.2880
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.8399
#> -1.6011
#> -0.5090
#> -2.2459
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.2411
#>  0.5102
#> -0.0886
#> -0.9739
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -1.3864
#> -0.3860
#>  0.3528
#> -0.0865
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.3530
#> -1.4395
#> -2.5759
#> -1.9576
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4639
#> -0.8822
#>  2.7892
#>  0.5885
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.7044
#>  0.4520
#> -0.5958
#> -1.5558
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.4735
#> -0.2830
#> -1.7554
#>  0.8171
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.7891
#>  0.6901
#> -1.3893
#>  0.0889
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.7774
#>  0.4797
#> -1.0913
#> -0.5842
#> [ CPUFloatType{4} ]
#> 
#> 
```
