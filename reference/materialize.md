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
#> -0.1376 -1.4496 -0.2042
#>  2.2617  0.6226 -0.9325
#> -0.7653 -0.1442  0.4354
#> -0.6718  0.9745 -0.9828
#> -0.7521  1.5415 -1.4919
#> -0.0998  0.0945 -0.4701
#> -0.5985 -0.0743  1.0002
#> -0.7081  0.1861 -0.5799
#> -0.5120 -0.2013 -0.2230
#>  1.2046  1.1828 -0.9459
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.1376
#> -1.4496
#> -0.2042
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  2.2617
#>  0.6226
#> -0.9325
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.7653
#> -0.1442
#>  0.4354
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6718
#>  0.9745
#> -0.9828
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.7521
#>  1.5415
#> -1.4919
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> 0.01 *
#> -9.9845
#>  9.4460
#> -47.0079
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.5985
#> -0.0743
#>  1.0002
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.7081
#>  0.1861
#> -0.5799
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.5120
#> -0.2013
#> -0.2230
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.2046
#>  1.1828
#> -0.9459
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.1376 -1.4496 -0.2042
#>  2.2617  0.6226 -0.9325
#> -0.7653 -0.1442  0.4354
#> -0.6718  0.9745 -0.9828
#> -0.7521  1.5415 -1.4919
#> -0.0998  0.0945 -0.4701
#> -0.5985 -0.0743  1.0002
#> -0.7081  0.1861 -0.5799
#> -0.5120 -0.2013 -0.2230
#>  1.2046  1.1828 -0.9459
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.5046 -1.3524 -0.7283  1.8251
#>  0.8644 -0.0547  1.1304  1.5233
#> -1.9413 -0.2160  0.4314  1.3274
#>  0.6988 -1.1334  0.7523  0.3125
#> -0.8385  1.6245  0.0775  0.0917
#>  0.1186  1.2582  1.3945 -0.0349
#> -1.5942  1.2985  1.3603 -1.1178
#> -0.1934 -0.0980  0.4713 -1.5077
#>  0.7507 -1.7573 -0.0274  2.3501
#>  0.0095 -0.1841 -1.2649  0.3106
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.1376
#> -1.4496
#> -0.2042
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  2.2617
#>  0.6226
#> -0.9325
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.7653
#> -0.1442
#>  0.4354
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6718
#>  0.9745
#> -0.9828
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.7521
#>  1.5415
#> -1.4919
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> 0.01 *
#> -9.9845
#>  9.4460
#> -47.0079
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.5985
#> -0.0743
#>  1.0002
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.7081
#>  0.1861
#> -0.5799
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.5120
#> -0.2013
#> -0.2230
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.2046
#>  1.1828
#> -0.9459
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.5046
#> -1.3524
#> -0.7283
#>  1.8251
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.8644
#> -0.0547
#>  1.1304
#>  1.5233
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.9413
#> -0.2160
#>  0.4314
#>  1.3274
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.6988
#> -1.1334
#>  0.7523
#>  0.3125
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.8385
#>  1.6245
#>  0.0775
#>  0.0917
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.1186
#>  1.2582
#>  1.3945
#> -0.0349
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.5942
#>  1.2985
#>  1.3603
#> -1.1178
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.1934
#> -0.0980
#>  0.4713
#> -1.5077
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.7507
#> -1.7573
#> -0.0274
#>  2.3501
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.0095
#> -0.1841
#> -1.2649
#>  0.3106
#> [ CPUFloatType{4} ]
#> 
#> 
```
