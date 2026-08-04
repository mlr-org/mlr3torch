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
#> -0.0028  1.6296  0.4374
#> -0.2287 -0.6491 -0.1104
#> -1.5279  0.1930 -1.3797
#>  1.4322  1.5807  1.6416
#>  0.2369  0.1747 -0.7079
#>  2.3734 -0.8067  0.0150
#>  1.0493 -0.5993  0.1721
#> -0.7960 -1.3827  0.5708
#> -1.7642  0.0448  1.5058
#>  0.4931 -0.6706  0.8319
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.0028
#>  1.6296
#>  0.4374
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.2287
#> -0.6491
#> -0.1104
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.5279
#>  0.1930
#> -1.3797
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  1.4322
#>  1.5807
#>  1.6416
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.2369
#>  0.1747
#> -0.7079
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  2.3734
#> -0.8067
#>  0.0150
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.0493
#> -0.5993
#>  0.1721
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.7960
#> -1.3827
#>  0.5708
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.7642
#>  0.0448
#>  1.5058
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4931
#> -0.6706
#>  0.8319
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.0028  1.6296  0.4374
#> -0.2287 -0.6491 -0.1104
#> -1.5279  0.1930 -1.3797
#>  1.4322  1.5807  1.6416
#>  0.2369  0.1747 -0.7079
#>  2.3734 -0.8067  0.0150
#>  1.0493 -0.5993  0.1721
#> -0.7960 -1.3827  0.5708
#> -1.7642  0.0448  1.5058
#>  0.4931 -0.6706  0.8319
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.5526  2.6949 -0.0639  0.4762
#> -0.7911  0.0899  1.0287 -0.0213
#>  1.9785 -1.3648 -1.0540 -1.9423
#> -0.2440 -0.4123 -0.7686 -0.0393
#>  1.0124  0.6818 -1.4501  0.5944
#>  0.1382 -0.2612  0.5902 -1.4626
#> -2.4494  0.4306 -2.3358  0.6775
#> -0.6567  1.5931  1.0535 -0.3117
#> -0.4887 -0.4654  1.7631 -1.2993
#>  0.9206  0.6366 -0.3342 -0.0586
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.0028
#>  1.6296
#>  0.4374
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.2287
#> -0.6491
#> -0.1104
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.5279
#>  0.1930
#> -1.3797
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  1.4322
#>  1.5807
#>  1.6416
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.2369
#>  0.1747
#> -0.7079
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  2.3734
#> -0.8067
#>  0.0150
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.0493
#> -0.5993
#>  0.1721
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.7960
#> -1.3827
#>  0.5708
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.7642
#>  0.0448
#>  1.5058
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4931
#> -0.6706
#>  0.8319
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.5526
#>  2.6949
#> -0.0639
#>  0.4762
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.7911
#>  0.0899
#>  1.0287
#> -0.0213
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.9785
#> -1.3648
#> -1.0540
#> -1.9423
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.2440
#> -0.4123
#> -0.7686
#> -0.0393
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.0124
#>  0.6818
#> -1.4501
#>  0.5944
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.1382
#> -0.2612
#>  0.5902
#> -1.4626
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -2.4494
#>  0.4306
#> -2.3358
#>  0.6775
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.6567
#>  1.5931
#>  1.0535
#> -0.3117
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.4887
#> -0.4654
#>  1.7631
#> -1.2993
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.9206
#>  0.6366
#> -0.3342
#> -0.0586
#> [ CPUFloatType{4} ]
#> 
#> 
```
