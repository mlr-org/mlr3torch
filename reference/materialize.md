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
#>  2.4568  1.0516 -0.6121
#>  0.2761 -0.9983 -0.1895
#>  1.9090  0.1342 -1.6492
#> -0.6837  0.4671  0.8151
#> -1.2810  0.4301 -0.1887
#> -0.2614 -1.1389 -0.0549
#> -0.2894 -0.7917 -0.6240
#> -0.0876 -0.7103 -0.4258
#>  1.4991 -0.6865  0.1089
#>  0.4665  0.7400  0.7912
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  2.4568
#>  1.0516
#> -0.6121
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.2761
#> -0.9983
#> -0.1895
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.9090
#>  0.1342
#> -1.6492
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6837
#>  0.4671
#>  0.8151
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.2810
#>  0.4301
#> -0.1887
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.2614
#> -1.1389
#> -0.0549
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.2894
#> -0.7917
#> -0.6240
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> 0.01 *
#> -8.7601
#> -71.0312
#> -42.5771
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.4991
#> -0.6865
#>  0.1089
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4665
#>  0.7400
#>  0.7912
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  2.4568  1.0516 -0.6121
#>  0.2761 -0.9983 -0.1895
#>  1.9090  0.1342 -1.6492
#> -0.6837  0.4671  0.8151
#> -1.2810  0.4301 -0.1887
#> -0.2614 -1.1389 -0.0549
#> -0.2894 -0.7917 -0.6240
#> -0.0876 -0.7103 -0.4258
#>  1.4991 -0.6865  0.1089
#>  0.4665  0.7400  0.7912
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.3126 -0.6971  0.4446 -1.5016
#> -0.4383 -0.4958 -0.5675 -1.0642
#>  0.5185 -1.7864 -0.9511 -1.4726
#> -0.1105  0.2763 -1.6143  1.4368
#>  1.3593  0.1711  0.1541 -0.2266
#>  0.4930  0.7958 -0.3975 -3.2543
#> -0.5642 -1.6493 -0.9212  0.7612
#>  0.0471 -0.8589  0.1239 -0.3530
#>  0.4819  0.1168  0.4746  0.8318
#>  0.3691  0.8970  0.2003  0.0464
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  2.4568
#>  1.0516
#> -0.6121
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.2761
#> -0.9983
#> -0.1895
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.9090
#>  0.1342
#> -1.6492
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6837
#>  0.4671
#>  0.8151
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.2810
#>  0.4301
#> -0.1887
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.2614
#> -1.1389
#> -0.0549
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.2894
#> -0.7917
#> -0.6240
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> 0.01 *
#> -8.7601
#> -71.0312
#> -42.5771
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.4991
#> -0.6865
#>  0.1089
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4665
#>  0.7400
#>  0.7912
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.3126
#> -0.6971
#>  0.4446
#> -1.5016
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.4383
#> -0.4958
#> -0.5675
#> -1.0642
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5185
#> -1.7864
#> -0.9511
#> -1.4726
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.1105
#>  0.2763
#> -1.6143
#>  1.4368
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.3593
#>  0.1711
#>  0.1541
#> -0.2266
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4930
#>  0.7958
#> -0.3975
#> -3.2543
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.5642
#> -1.6493
#> -0.9212
#>  0.7612
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.0471
#> -0.8589
#>  0.1239
#> -0.3530
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.4819
#>  0.1168
#>  0.4746
#>  0.8318
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.3691
#>  0.8970
#>  0.2003
#>  0.0464
#> [ CPUFloatType{4} ]
#> 
#> 
```
