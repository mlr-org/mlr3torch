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
#>  0.1688 -0.3524  0.6344
#>  0.2791 -1.5378  0.7289
#>  0.4784 -0.3086 -0.8675
#>  0.1779  0.4412  0.7848
#> -0.6213 -0.5237 -0.8676
#>  0.5768  0.2822  0.2859
#> -0.1057 -0.2558 -0.7513
#>  0.6082  0.7798  0.4257
#> -0.2507 -1.6690 -1.5447
#>  0.2562  1.3777  0.0530
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.1688
#> -0.3524
#>  0.6344
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.2791
#> -1.5378
#>  0.7289
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.4784
#> -0.3086
#> -0.8675
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.1779
#>  0.4412
#>  0.7848
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.6213
#> -0.5237
#> -0.8676
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.5768
#>  0.2822
#>  0.2859
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.1057
#> -0.2558
#> -0.7513
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.6082
#>  0.7798
#>  0.4257
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.2507
#> -1.6690
#> -1.5447
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.2562
#>  1.3777
#>  0.0530
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.1688 -0.3524  0.6344
#>  0.2791 -1.5378  0.7289
#>  0.4784 -0.3086 -0.8675
#>  0.1779  0.4412  0.7848
#> -0.6213 -0.5237 -0.8676
#>  0.5768  0.2822  0.2859
#> -0.1057 -0.2558 -0.7513
#>  0.6082  0.7798  0.4257
#> -0.2507 -1.6690 -1.5447
#>  0.2562  1.3777  0.0530
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2088 -0.8319 -0.2201 -1.2380
#>  0.1598 -0.0243 -0.0247 -1.7070
#>  0.0492 -0.0483  0.0266 -0.3942
#>  0.0985 -0.0495  0.2748  0.2543
#>  0.7461 -0.2640  0.5059 -0.1363
#> -0.5765  0.5455 -0.9782 -0.5140
#>  0.3515  0.3839  2.1589  0.9503
#>  0.9004  0.9719 -0.6611 -0.7708
#> -0.6126  1.1825  1.0074 -0.7365
#>  0.8273 -0.1296  0.4352  2.5884
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.1688
#> -0.3524
#>  0.6344
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.2791
#> -1.5378
#>  0.7289
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.4784
#> -0.3086
#> -0.8675
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.1779
#>  0.4412
#>  0.7848
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.6213
#> -0.5237
#> -0.8676
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.5768
#>  0.2822
#>  0.2859
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.1057
#> -0.2558
#> -0.7513
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.6082
#>  0.7798
#>  0.4257
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.2507
#> -1.6690
#> -1.5447
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.2562
#>  1.3777
#>  0.0530
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2088
#> -0.8319
#> -0.2201
#> -1.2380
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.1598
#> -0.0243
#> -0.0247
#> -1.7070
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.0492
#> -0.0483
#>  0.0266
#> -0.3942
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.0985
#> -0.0495
#>  0.2748
#>  0.2543
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.7461
#> -0.2640
#>  0.5059
#> -0.1363
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.5765
#>  0.5455
#> -0.9782
#> -0.5140
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.3515
#>  0.3839
#>  2.1589
#>  0.9503
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.9004
#>  0.9719
#> -0.6611
#> -0.7708
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.6126
#>  1.1825
#>  1.0074
#> -0.7365
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.8273
#> -0.1296
#>  0.4352
#>  2.5884
#> [ CPUFloatType{4} ]
#> 
#> 
```
