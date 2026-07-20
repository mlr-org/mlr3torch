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
#>  1.0927 -0.1747  0.9811
#>  0.3541  0.6676  1.4485
#> -0.5489  0.4032 -1.6980
#> -0.0284  1.7651  0.4877
#> -0.0140  0.8336  0.1808
#>  0.5974 -0.3915 -0.5563
#>  1.1276  0.2116  0.6638
#>  2.6562  0.5505  0.3343
#> -1.0430  0.2526 -1.8832
#> -0.3419 -0.0214  0.6593
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.0927
#> -0.1747
#>  0.9811
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.3541
#>  0.6676
#>  1.4485
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.5489
#>  0.4032
#> -1.6980
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.0284
#>  1.7651
#>  0.4877
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.0140
#>  0.8336
#>  0.1808
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.5974
#> -0.3915
#> -0.5563
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.1276
#>  0.2116
#>  0.6638
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  2.6562
#>  0.5505
#>  0.3343
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.0430
#>  0.2526
#> -1.8832
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.3419
#> -0.0214
#>  0.6593
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.0927 -0.1747  0.9811
#>  0.3541  0.6676  1.4485
#> -0.5489  0.4032 -1.6980
#> -0.0284  1.7651  0.4877
#> -0.0140  0.8336  0.1808
#>  0.5974 -0.3915 -0.5563
#>  1.1276  0.2116  0.6638
#>  2.6562  0.5505  0.3343
#> -1.0430  0.2526 -1.8832
#> -0.3419 -0.0214  0.6593
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.7766 -0.8941 -0.0695  0.3444
#> -0.7506  1.0279 -0.1556 -1.5792
#>  0.0996 -1.7343  0.5081  1.3159
#> -0.0670  0.7805 -0.1522 -2.0933
#> -0.2786  0.4746 -0.1963  0.4866
#> -2.1676 -0.1521  0.8101  0.8347
#>  0.2450 -1.0030  1.3159 -0.4506
#> -1.7705  1.5644  0.0040  1.4950
#>  0.2509 -0.7498 -0.6917  0.9876
#> -0.1896  0.4112  1.7561 -0.3427
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.0927
#> -0.1747
#>  0.9811
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.3541
#>  0.6676
#>  1.4485
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.5489
#>  0.4032
#> -1.6980
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.0284
#>  1.7651
#>  0.4877
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.0140
#>  0.8336
#>  0.1808
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.5974
#> -0.3915
#> -0.5563
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.1276
#>  0.2116
#>  0.6638
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  2.6562
#>  0.5505
#>  0.3343
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.0430
#>  0.2526
#> -1.8832
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.3419
#> -0.0214
#>  0.6593
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.7766
#> -0.8941
#> -0.0695
#>  0.3444
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.7506
#>  1.0279
#> -0.1556
#> -1.5792
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.0996
#> -1.7343
#>  0.5081
#>  1.3159
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.0670
#>  0.7805
#> -0.1522
#> -2.0933
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.2786
#>  0.4746
#> -0.1963
#>  0.4866
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -2.1676
#> -0.1521
#>  0.8101
#>  0.8347
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.2450
#> -1.0030
#>  1.3159
#> -0.4506
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.7705
#>  1.5644
#>  0.0040
#>  1.4950
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.2509
#> -0.7498
#> -0.6917
#>  0.9876
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.1896
#>  0.4112
#>  1.7561
#> -0.3427
#> [ CPUFloatType{4} ]
#> 
#> 
```
