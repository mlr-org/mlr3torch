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
#>  0.1572 -0.1862  0.9653
#>  0.6766 -1.0777 -0.5574
#> -0.2593 -0.5987  0.8528
#> -0.6782  0.2726 -1.4050
#> -0.0257  0.2379  0.7741
#> -1.2831  0.1585 -0.7927
#> -0.8867 -0.0951 -2.1687
#>  1.3722  0.5888  0.0024
#>  0.1422  0.3142  0.2759
#>  0.4617  1.6267 -1.0592
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.1572
#> -0.1862
#>  0.9653
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.6766
#> -1.0777
#> -0.5574
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.2593
#> -0.5987
#>  0.8528
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.6782
#>  0.2726
#> -1.4050
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.0257
#>  0.2379
#>  0.7741
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -1.2831
#>  0.1585
#> -0.7927
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.8867
#> -0.0951
#> -2.1687
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.3722
#>  0.5888
#>  0.0024
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.1422
#>  0.3142
#>  0.2759
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4617
#>  1.6267
#> -1.0592
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.1572 -0.1862  0.9653
#>  0.6766 -1.0777 -0.5574
#> -0.2593 -0.5987  0.8528
#> -0.6782  0.2726 -1.4050
#> -0.0257  0.2379  0.7741
#> -1.2831  0.1585 -0.7927
#> -0.8867 -0.0951 -2.1687
#>  1.3722  0.5888  0.0024
#>  0.1422  0.3142  0.2759
#>  0.4617  1.6267 -1.0592
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.0977 -0.9840  0.7736 -0.6235
#>  1.0158  0.3392  1.3174  1.2433
#>  0.8230  1.6068  1.3039  0.4258
#>  0.2782  0.4576 -1.3199 -0.1118
#> -1.6451  0.9528  0.7079  0.8511
#> -1.4052  1.3003  1.1886 -0.5403
#>  0.5103 -0.9378 -0.7151  0.0285
#>  1.2453 -0.2739  0.2127  1.1654
#> -1.2927  0.7216 -1.4850  0.1569
#> -0.0055 -0.7681 -0.8301 -0.0199
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.1572
#> -0.1862
#>  0.9653
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.6766
#> -1.0777
#> -0.5574
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.2593
#> -0.5987
#>  0.8528
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.6782
#>  0.2726
#> -1.4050
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.0257
#>  0.2379
#>  0.7741
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -1.2831
#>  0.1585
#> -0.7927
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.8867
#> -0.0951
#> -2.1687
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.3722
#>  0.5888
#>  0.0024
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.1422
#>  0.3142
#>  0.2759
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4617
#>  1.6267
#> -1.0592
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.0977
#> -0.9840
#>  0.7736
#> -0.6235
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.0158
#>  0.3392
#>  1.3174
#>  1.2433
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.8230
#>  1.6068
#>  1.3039
#>  0.4258
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2782
#>  0.4576
#> -1.3199
#> -0.1118
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.6451
#>  0.9528
#>  0.7079
#>  0.8511
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.4052
#>  1.3003
#>  1.1886
#> -0.5403
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.5103
#> -0.9378
#> -0.7151
#>  0.0285
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.2453
#> -0.2739
#>  0.2127
#>  1.1654
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.2927
#>  0.7216
#> -1.4850
#>  0.1569
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.0055
#> -0.7681
#> -0.8301
#> -0.0199
#> [ CPUFloatType{4} ]
#> 
#> 
```
