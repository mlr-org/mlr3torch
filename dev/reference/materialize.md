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
#> -0.5596  0.2989 -0.6992
#>  0.2863  1.8108  0.6243
#>  0.6931 -1.1461 -0.7378
#> -0.4639  0.1389  1.0796
#> -1.8056  1.1916 -0.5017
#>  0.1472 -0.0935  1.8966
#>  0.6154 -0.7977  1.2984
#>  1.0915 -1.0085 -0.1097
#> -0.0776  0.2747  1.0510
#> -0.1578  0.6112  1.2673
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.5596
#>  0.2989
#> -0.6992
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.2863
#>  1.8108
#>  0.6243
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.6931
#> -1.1461
#> -0.7378
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.4639
#>  0.1389
#>  1.0796
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.8056
#>  1.1916
#> -0.5017
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.1472
#> -0.0935
#>  1.8966
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.6154
#> -0.7977
#>  1.2984
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.0915
#> -1.0085
#> -0.1097
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.0776
#>  0.2747
#>  1.0510
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.1578
#>  0.6112
#>  1.2673
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.5596  0.2989 -0.6992
#>  0.2863  1.8108  0.6243
#>  0.6931 -1.1461 -0.7378
#> -0.4639  0.1389  1.0796
#> -1.8056  1.1916 -0.5017
#>  0.1472 -0.0935  1.8966
#>  0.6154 -0.7977  1.2984
#>  1.0915 -1.0085 -0.1097
#> -0.0776  0.2747  1.0510
#> -0.1578  0.6112  1.2673
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.9529  0.4856 -1.6786  0.2060
#> -0.3989  0.7514 -0.0451  0.4483
#>  0.6548 -0.0393 -0.3391 -1.5001
#>  0.2064 -0.5345  0.3333  0.7413
#> -0.1164  0.8336  0.9004  0.3914
#>  0.8218  0.8176  0.4242  0.0780
#> -0.7701  0.9004 -0.9035  2.1012
#> -0.7160  0.1664 -0.1011  2.0801
#> -1.6004 -0.4684 -0.1892  0.7419
#> -0.1137 -1.1472 -0.2620 -0.8667
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.5596
#>  0.2989
#> -0.6992
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.2863
#>  1.8108
#>  0.6243
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.6931
#> -1.1461
#> -0.7378
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.4639
#>  0.1389
#>  1.0796
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.8056
#>  1.1916
#> -0.5017
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.1472
#> -0.0935
#>  1.8966
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.6154
#> -0.7977
#>  1.2984
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.0915
#> -1.0085
#> -0.1097
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.0776
#>  0.2747
#>  1.0510
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.1578
#>  0.6112
#>  1.2673
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.9529
#>  0.4856
#> -1.6786
#>  0.2060
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.3989
#>  0.7514
#> -0.0451
#>  0.4483
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.6548
#> -0.0393
#> -0.3391
#> -1.5001
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2064
#> -0.5345
#>  0.3333
#>  0.7413
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.1164
#>  0.8336
#>  0.9004
#>  0.3914
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.8218
#>  0.8176
#>  0.4242
#>  0.0780
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.7701
#>  0.9004
#> -0.9035
#>  2.1012
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.7160
#>  0.1664
#> -0.1011
#>  2.0801
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.6004
#> -0.4684
#> -0.1892
#>  0.7419
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.1137
#> -1.1472
#> -0.2620
#> -0.8667
#> [ CPUFloatType{4} ]
#> 
#> 
```
