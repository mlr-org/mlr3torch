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

  (`character(1)` or [`hashtab()`](https://rdrr.io/r/utils/hashtab.html)
  or `NULL`)  
  Optional cache for (intermediate) materialization results. Per
  default, caching will be enabled when the same dataset or data
  descriptor (with different output pointer) is used for more than one
  lazy tensor column.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)s
or a
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html))

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

For this reason it is possible to provide a cache, which is a
[`hashtab()`](https://rdrr.io/r/utils/hashtab.html). The key for a) is
`list(dataset, indices)`, the key for b) is
`list(indices, dataset, graph, input_map)`. The dataset and the graph go
into the key as the objects themselves, so keys are compared with
[`identical()`](https://rdrr.io/r/base/identical.html) rather than being
digested into a string, and two different keys can never share an entry.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#>  0.0116 -0.1686  1.0710
#> -0.6801 -1.6071  0.5040
#>  0.4988 -0.9801  0.2618
#>  0.3454  0.2855  1.1944
#>  2.0975 -0.3290  0.5945
#>  1.3702 -0.8095 -0.6643
#>  0.0904  0.1303 -0.7842
#>  0.1681  0.6266 -0.0524
#> -0.2820 -0.0609  0.9416
#> -1.0806  0.3510 -0.6813
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.0116
#> -0.1686
#>  1.0710
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.6801
#> -1.6071
#>  0.5040
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.4988
#> -0.9801
#>  0.2618
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.3454
#>  0.2855
#>  1.1944
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  2.0975
#> -0.3290
#>  0.5945
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.3702
#> -0.8095
#> -0.6643
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.0904
#>  0.1303
#> -0.7842
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.1681
#>  0.6266
#> -0.0524
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.2820
#> -0.0609
#>  0.9416
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.0806
#>  0.3510
#> -0.6813
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.0116 -0.1686  1.0710
#> -0.6801 -1.6071  0.5040
#>  0.4988 -0.9801  0.2618
#>  0.3454  0.2855  1.1944
#>  2.0975 -0.3290  0.5945
#>  1.3702 -0.8095 -0.6643
#>  0.0904  0.1303 -0.7842
#>  0.1681  0.6266 -0.0524
#> -0.2820 -0.0609  0.9416
#> -1.0806  0.3510 -0.6813
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.5959 -0.2156 -0.0057  0.1472
#>  0.2596 -1.1499  0.9905  1.8599
#> -0.7539  0.7214 -0.4738 -1.1442
#> -0.4703 -0.3824 -1.5519 -0.4678
#>  0.7113 -0.2991  1.3950  1.1588
#>  0.7058 -1.4547 -0.2623 -0.7930
#> -0.7080  0.1201 -1.9445 -0.2086
#> -0.8031  0.7661 -0.4190 -0.3884
#>  1.8033 -0.4537 -1.2297 -0.0875
#>  1.9847 -1.9368 -0.0960  1.0159
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.0116
#> -0.1686
#>  1.0710
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.6801
#> -1.6071
#>  0.5040
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.4988
#> -0.9801
#>  0.2618
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.3454
#>  0.2855
#>  1.1944
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  2.0975
#> -0.3290
#>  0.5945
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.3702
#> -0.8095
#> -0.6643
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.0904
#>  0.1303
#> -0.7842
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.1681
#>  0.6266
#> -0.0524
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.2820
#> -0.0609
#>  0.9416
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.0806
#>  0.3510
#> -0.6813
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.5959
#> -0.2156
#> -0.0057
#>  0.1472
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.2596
#> -1.1499
#>  0.9905
#>  1.8599
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.7539
#>  0.7214
#> -0.4738
#> -1.1442
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.4703
#> -0.3824
#> -1.5519
#> -0.4678
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.7113
#> -0.2991
#>  1.3950
#>  1.1588
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.7058
#> -1.4547
#> -0.2623
#> -0.7930
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.7080
#>  0.1201
#> -1.9445
#> -0.2086
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.8031
#>  0.7661
#> -0.4190
#> -0.3884
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.8033
#> -0.4537
#> -1.2297
#> -0.0875
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.9847
#> -1.9368
#> -0.0960
#>  1.0159
#> [ CPUFloatType{4} ]
#> 
#> 
```
