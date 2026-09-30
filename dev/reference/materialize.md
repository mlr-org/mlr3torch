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
#> -0.8346 -0.4806 -0.7892
#> -2.3053  0.3374 -0.0660
#> -0.7134  0.4342 -0.6195
#> -0.1699  0.6419 -0.8134
#> -0.2638 -3.1809  0.4519
#>  0.7884  0.9032  2.0956
#>  2.0572  0.3568 -1.7600
#> -0.9147 -0.2800 -0.3734
#>  0.1052 -0.3304  1.0913
#>  0.6509  0.2893 -1.1003
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.8346
#> -0.4806
#> -0.7892
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -2.3053
#>  0.3374
#> -0.0660
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.7134
#>  0.4342
#> -0.6195
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.1699
#>  0.6419
#> -0.8134
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2638
#> -3.1809
#>  0.4519
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.7884
#>  0.9032
#>  2.0956
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  2.0572
#>  0.3568
#> -1.7600
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.9147
#> -0.2800
#> -0.3734
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.1052
#> -0.3304
#>  1.0913
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.6509
#>  0.2893
#> -1.1003
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.8346 -0.4806 -0.7892
#> -2.3053  0.3374 -0.0660
#> -0.7134  0.4342 -0.6195
#> -0.1699  0.6419 -0.8134
#> -0.2638 -3.1809  0.4519
#>  0.7884  0.9032  2.0956
#>  2.0572  0.3568 -1.7600
#> -0.9147 -0.2800 -0.3734
#>  0.1052 -0.3304  1.0913
#>  0.6509  0.2893 -1.1003
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  2.2418 -1.0943 -1.2482 -0.4824
#>  0.7801 -0.5008 -0.1048  0.3537
#> -0.2348  1.7211 -0.1323  0.4435
#> -0.8561 -0.1986  0.0285  0.1978
#> -0.6432  0.0393 -0.0147 -0.7277
#>  0.4601 -1.5636  1.5509  0.0134
#>  0.3275  0.8436  1.0405  0.1937
#> -1.1039 -0.8660 -0.0730  0.9251
#> -1.9658  0.3190  0.9645 -0.5277
#> -1.0537 -0.3336  1.5911 -0.8828
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.8346
#> -0.4806
#> -0.7892
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -2.3053
#>  0.3374
#> -0.0660
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.7134
#>  0.4342
#> -0.6195
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.1699
#>  0.6419
#> -0.8134
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2638
#> -3.1809
#>  0.4519
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.7884
#>  0.9032
#>  2.0956
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  2.0572
#>  0.3568
#> -1.7600
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.9147
#> -0.2800
#> -0.3734
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.1052
#> -0.3304
#>  1.0913
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.6509
#>  0.2893
#> -1.1003
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  2.2418
#> -1.0943
#> -1.2482
#> -0.4824
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.7801
#> -0.5008
#> -0.1048
#>  0.3537
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.2348
#>  1.7211
#> -0.1323
#>  0.4435
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.8561
#> -0.1986
#>  0.0285
#>  0.1978
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.6432
#>  0.0393
#> -0.0147
#> -0.7277
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.4601
#> -1.5636
#>  1.5509
#>  0.0134
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.3275
#>  0.8436
#>  1.0405
#>  0.1937
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.1039
#> -0.8660
#> -0.0730
#>  0.9251
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -1.9658
#>  0.3190
#>  0.9645
#> -0.5277
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.0537
#> -0.3336
#>  1.5911
#> -0.8828
#> [ CPUFloatType{4} ]
#> 
#> 
```
