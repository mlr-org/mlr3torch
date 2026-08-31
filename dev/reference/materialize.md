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
#>  0.4281  0.1932  0.2554
#> -1.7500  0.4130 -0.2940
#>  0.2581  1.4857 -1.2949
#>  0.5263 -1.7081 -0.0314
#> -0.0183 -0.8333  1.3986
#>  0.1851 -0.0037 -0.4172
#>  0.2381  0.9334 -0.3862
#> -0.4996 -0.4315  0.1302
#>  0.4866  0.8742 -0.1270
#>  0.6347 -0.3746 -1.2360
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.4281
#>  0.1932
#>  0.2554
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.7500
#>  0.4130
#> -0.2940
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.2581
#>  1.4857
#> -1.2949
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.5263
#> -1.7081
#> -0.0314
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.0183
#> -0.8333
#>  1.3986
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.1851
#> -0.0037
#> -0.4172
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.2381
#>  0.9334
#> -0.3862
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.4996
#> -0.4315
#>  0.1302
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.4866
#>  0.8742
#> -0.1270
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.6347
#> -0.3746
#> -1.2360
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.4281  0.1932  0.2554
#> -1.7500  0.4130 -0.2940
#>  0.2581  1.4857 -1.2949
#>  0.5263 -1.7081 -0.0314
#> -0.0183 -0.8333  1.3986
#>  0.1851 -0.0037 -0.4172
#>  0.2381  0.9334 -0.3862
#> -0.4996 -0.4315  0.1302
#>  0.4866  0.8742 -0.1270
#>  0.6347 -0.3746 -1.2360
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.1363 -0.4385 -0.1197 -2.4644
#>  1.8879  0.1076  2.1047  0.0986
#> -0.6907 -0.0913  0.7492 -0.9644
#> -0.4747  0.8067 -1.5944 -0.5265
#>  0.6898  1.8698 -1.2373 -0.7388
#>  0.2384  1.0757  1.7531 -0.0399
#> -1.3241  0.0519 -1.1156  0.6062
#>  0.4522  0.5966  1.0498  0.6847
#>  0.9593  0.3920 -1.9442 -0.2163
#> -0.1916  3.3278  1.2812 -0.1936
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.4281
#>  0.1932
#>  0.2554
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.7500
#>  0.4130
#> -0.2940
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.2581
#>  1.4857
#> -1.2949
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.5263
#> -1.7081
#> -0.0314
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.0183
#> -0.8333
#>  1.3986
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.1851
#> -0.0037
#> -0.4172
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.2381
#>  0.9334
#> -0.3862
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.4996
#> -0.4315
#>  0.1302
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.4866
#>  0.8742
#> -0.1270
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.6347
#> -0.3746
#> -1.2360
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.1363
#> -0.4385
#> -0.1197
#> -2.4644
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.8879
#>  0.1076
#>  2.1047
#>  0.0986
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.6907
#> -0.0913
#>  0.7492
#> -0.9644
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.4747
#>  0.8067
#> -1.5944
#> -0.5265
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.6898
#>  1.8698
#> -1.2373
#> -0.7388
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.2384
#>  1.0757
#>  1.7531
#> -0.0399
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.3241
#>  0.0519
#> -1.1156
#>  0.6062
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4522
#>  0.5966
#>  1.0498
#>  0.6847
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.9593
#>  0.3920
#> -1.9442
#> -0.2163
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.1916
#>  3.3278
#>  1.2812
#> -0.1936
#> [ CPUFloatType{4} ]
#> 
#> 
```
