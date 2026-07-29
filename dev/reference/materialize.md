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
#> -1.1337 -2.0737  1.7357
#> -0.3843  0.8726  0.3589
#>  0.1222 -0.1142  0.0538
#> -1.8960  0.0402  0.3530
#> -0.2787 -0.5239 -1.1214
#>  0.5365 -0.5130 -1.5617
#> -0.3035  0.4977 -1.3345
#>  0.2807 -0.1477  0.8364
#> -0.7671 -0.2097 -2.7674
#>  0.1574 -0.7247 -0.0483
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.1337
#> -2.0737
#>  1.7357
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3843
#>  0.8726
#>  0.3589
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.1222
#> -0.1142
#>  0.0538
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.8960
#>  0.0402
#>  0.3530
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2787
#> -0.5239
#> -1.1214
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.5365
#> -0.5130
#> -1.5617
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.3035
#>  0.4977
#> -1.3345
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.2807
#> -0.1477
#>  0.8364
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.7671
#> -0.2097
#> -2.7674
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.1574
#> -0.7247
#> -0.0483
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.1337 -2.0737  1.7357
#> -0.3843  0.8726  0.3589
#>  0.1222 -0.1142  0.0538
#> -1.8960  0.0402  0.3530
#> -0.2787 -0.5239 -1.1214
#>  0.5365 -0.5130 -1.5617
#> -0.3035  0.4977 -1.3345
#>  0.2807 -0.1477  0.8364
#> -0.7671 -0.2097 -2.7674
#>  0.1574 -0.7247 -0.0483
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.5692  1.3279 -1.4389 -0.8627
#> -0.0930  0.6219  0.7821  0.2891
#> -0.0180 -1.2425 -1.4915  1.3245
#> -0.7465 -1.3961 -1.0504  0.0287
#>  0.4219 -0.1721  0.6696  0.6622
#> -0.6638 -0.8870 -0.8487  0.5364
#> -0.0127 -0.6097 -0.1866 -1.0009
#>  1.6740 -0.4403  1.8519  0.4226
#>  0.2764  0.6159  0.0924  1.2003
#> -1.5716  0.2032  0.5000  1.6275
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.1337
#> -2.0737
#>  1.7357
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3843
#>  0.8726
#>  0.3589
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.1222
#> -0.1142
#>  0.0538
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.8960
#>  0.0402
#>  0.3530
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2787
#> -0.5239
#> -1.1214
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.5365
#> -0.5130
#> -1.5617
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.3035
#>  0.4977
#> -1.3345
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.2807
#> -0.1477
#>  0.8364
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.7671
#> -0.2097
#> -2.7674
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.1574
#> -0.7247
#> -0.0483
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.5692
#>  1.3279
#> -1.4389
#> -0.8627
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.0930
#>  0.6219
#>  0.7821
#>  0.2891
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.0180
#> -1.2425
#> -1.4915
#>  1.3245
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.7465
#> -1.3961
#> -1.0504
#>  0.0287
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.4219
#> -0.1721
#>  0.6696
#>  0.6622
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.6638
#> -0.8870
#> -0.8487
#>  0.5364
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.0127
#> -0.6097
#> -0.1866
#> -1.0009
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.6740
#> -0.4403
#>  1.8519
#>  0.4226
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.2764
#>  0.6159
#>  0.0924
#>  1.2003
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.5716
#>  0.2032
#>  0.5000
#>  1.6275
#> [ CPUFloatType{4} ]
#> 
#> 
```
