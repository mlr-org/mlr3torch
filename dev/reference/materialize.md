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
#>  1.4318  1.5705 -1.7824
#>  0.3214  1.2965  0.6719
#> -1.8914  0.0307  1.2595
#> -1.0034 -0.8811 -0.7327
#> -0.2513 -1.3940  0.7492
#> -0.9130 -0.4014 -0.8623
#>  0.1762  0.6581 -1.1847
#>  0.3051  0.0239 -1.0506
#> -0.6201 -0.4884  0.6687
#>  0.4933 -0.2368  0.4783
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.4318
#>  1.5705
#> -1.7824
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.3214
#>  1.2965
#>  0.6719
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.8914
#>  0.0307
#>  1.2595
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.0034
#> -0.8811
#> -0.7327
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2513
#> -1.3940
#>  0.7492
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.9130
#> -0.4014
#> -0.8623
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.1762
#>  0.6581
#> -1.1847
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.3051
#>  0.0239
#> -1.0506
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.6201
#> -0.4884
#>  0.6687
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.4933
#> -0.2368
#>  0.4783
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.4318  1.5705 -1.7824
#>  0.3214  1.2965  0.6719
#> -1.8914  0.0307  1.2595
#> -1.0034 -0.8811 -0.7327
#> -0.2513 -1.3940  0.7492
#> -0.9130 -0.4014 -0.8623
#>  0.1762  0.6581 -1.1847
#>  0.3051  0.0239 -1.0506
#> -0.6201 -0.4884  0.6687
#>  0.4933 -0.2368  0.4783
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.0497  0.6343 -0.1165 -1.1400
#> -0.6911  0.8125  1.9974  0.6803
#> -0.0747 -0.2295  1.3624 -2.3607
#> -0.0126 -0.3391  0.5836  0.3766
#>  0.0288 -0.0030 -0.5580  0.3568
#>  0.4873 -1.1584 -0.4886  0.6068
#> -1.2368  0.5284  1.2664  0.7721
#>  1.6829 -0.2201 -0.2492 -0.1680
#>  0.5668  0.4335  0.1621 -0.6476
#> -2.0464  0.2358  0.2861 -0.4785
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.4318
#>  1.5705
#> -1.7824
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.3214
#>  1.2965
#>  0.6719
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.8914
#>  0.0307
#>  1.2595
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.0034
#> -0.8811
#> -0.7327
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2513
#> -1.3940
#>  0.7492
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.9130
#> -0.4014
#> -0.8623
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.1762
#>  0.6581
#> -1.1847
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.3051
#>  0.0239
#> -1.0506
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.6201
#> -0.4884
#>  0.6687
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.4933
#> -0.2368
#>  0.4783
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.0497
#>  0.6343
#> -0.1165
#> -1.1400
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.6911
#>  0.8125
#>  1.9974
#>  0.6803
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.0747
#> -0.2295
#>  1.3624
#> -2.3607
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.0126
#> -0.3391
#>  0.5836
#>  0.3766
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.0288
#> -0.0030
#> -0.5580
#>  0.3568
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4873
#> -1.1584
#> -0.4886
#>  0.6068
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.2368
#>  0.5284
#>  1.2664
#>  0.7721
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.6829
#> -0.2201
#> -0.2492
#> -0.1680
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.5668
#>  0.4335
#>  0.1621
#> -0.6476
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -2.0464
#>  0.2358
#>  0.2861
#> -0.4785
#> [ CPUFloatType{4} ]
#> 
#> 
```
