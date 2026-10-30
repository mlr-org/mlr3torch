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
#> -1.2915  0.7739  0.2913
#> -0.4707  0.7415  1.1826
#>  0.5332 -2.3390  0.3779
#>  0.7892 -1.0851 -0.0017
#> -1.4018 -1.3830  0.2254
#>  0.6065 -0.2919  0.5107
#>  0.8281 -1.1650  1.7550
#> -0.9515 -2.3409 -0.0547
#>  0.5146  0.3181 -1.4317
#> -1.8621 -0.7988  0.3510
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.2915
#>  0.7739
#>  0.2913
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.4707
#>  0.7415
#>  1.1826
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.5332
#> -2.3390
#>  0.3779
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7892
#> -1.0851
#> -0.0017
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.4018
#> -1.3830
#>  0.2254
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.6065
#> -0.2919
#>  0.5107
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.8281
#> -1.1650
#>  1.7550
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.9515
#> -2.3409
#> -0.0547
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.5146
#>  0.3181
#> -1.4317
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.8621
#> -0.7988
#>  0.3510
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.2915  0.7739  0.2913
#> -0.4707  0.7415  1.1826
#>  0.5332 -2.3390  0.3779
#>  0.7892 -1.0851 -0.0017
#> -1.4018 -1.3830  0.2254
#>  0.6065 -0.2919  0.5107
#>  0.8281 -1.1650  1.7550
#> -0.9515 -2.3409 -0.0547
#>  0.5146  0.3181 -1.4317
#> -1.8621 -0.7988  0.3510
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.8009 -1.0849  0.4215 -0.2999
#>  0.3420 -0.9680  1.6608  0.4863
#>  2.1048  0.9346  0.4704  0.1747
#>  0.0387 -0.6359 -0.0272 -0.2150
#>  1.4860 -0.1515 -0.0057  0.2940
#>  0.0751 -0.3440 -2.0824  1.2062
#> -0.2711  1.3294  0.9598 -0.5385
#>  1.4970  0.9526  1.0714  1.0542
#>  0.2738  0.8295 -2.1298  1.2379
#> -1.5229 -0.8376 -1.2782  0.2276
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.2915
#>  0.7739
#>  0.2913
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.4707
#>  0.7415
#>  1.1826
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.5332
#> -2.3390
#>  0.3779
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7892
#> -1.0851
#> -0.0017
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.4018
#> -1.3830
#>  0.2254
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.6065
#> -0.2919
#>  0.5107
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.8281
#> -1.1650
#>  1.7550
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.9515
#> -2.3409
#> -0.0547
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.5146
#>  0.3181
#> -1.4317
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.8621
#> -0.7988
#>  0.3510
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.8009
#> -1.0849
#>  0.4215
#> -0.2999
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.3420
#> -0.9680
#>  1.6608
#>  0.4863
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  2.1048
#>  0.9346
#>  0.4704
#>  0.1747
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.0387
#> -0.6359
#> -0.0272
#> -0.2150
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.4860
#> -0.1515
#> -0.0057
#>  0.2940
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.0751
#> -0.3440
#> -2.0824
#>  1.2062
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.2711
#>  1.3294
#>  0.9598
#> -0.5385
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.4970
#>  0.9526
#>  1.0714
#>  1.0542
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.2738
#>  0.8295
#> -2.1298
#>  1.2379
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.5229
#> -0.8376
#> -1.2782
#>  0.2276
#> [ CPUFloatType{4} ]
#> 
#> 
```
