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
#>  1.2298  1.3297 -0.2405
#> -0.8028 -1.1381  0.9022
#> -0.3355 -1.9173 -1.4886
#>  0.1132 -0.8117 -0.3362
#> -1.5763 -0.6182  1.2432
#> -1.6112 -0.5381 -1.0392
#>  0.1583  0.2419 -2.0671
#> -1.1396  0.9805  0.6740
#> -0.3306  1.1924  0.6999
#> -0.0031 -0.5660  0.0235
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.2298
#>  1.3297
#> -0.2405
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.8028
#> -1.1381
#>  0.9022
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.3355
#> -1.9173
#> -1.4886
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.1132
#> -0.8117
#> -0.3362
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.5763
#> -0.6182
#>  1.2432
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -1.6112
#> -0.5381
#> -1.0392
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.1583
#>  0.2419
#> -2.0671
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.1396
#>  0.9805
#>  0.6740
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.3306
#>  1.1924
#>  0.6999
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.0031
#> -0.5660
#>  0.0235
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.2298  1.3297 -0.2405
#> -0.8028 -1.1381  0.9022
#> -0.3355 -1.9173 -1.4886
#>  0.1132 -0.8117 -0.3362
#> -1.5763 -0.6182  1.2432
#> -1.6112 -0.5381 -1.0392
#>  0.1583  0.2419 -2.0671
#> -1.1396  0.9805  0.6740
#> -0.3306  1.1924  0.6999
#> -0.0031 -0.5660  0.0235
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2436 -2.2715 -1.3631 -0.3225
#> -0.8512  0.0334  1.0734  2.2791
#>  0.2775  2.0649  0.7591 -0.0965
#> -0.4159  0.4118 -0.7226  1.0895
#>  0.8616  0.9163  1.0176 -0.8928
#>  0.5088 -0.1886 -0.9851 -1.1011
#> -0.0726 -0.8237 -0.3192 -0.7509
#> -1.6198 -1.2828 -0.3089  0.2668
#>  0.9968  1.7800 -0.0981  0.6959
#>  1.1826  0.2707  0.2966 -0.5139
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.2298
#>  1.3297
#> -0.2405
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.8028
#> -1.1381
#>  0.9022
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.3355
#> -1.9173
#> -1.4886
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.1132
#> -0.8117
#> -0.3362
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.5763
#> -0.6182
#>  1.2432
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -1.6112
#> -0.5381
#> -1.0392
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.1583
#>  0.2419
#> -2.0671
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.1396
#>  0.9805
#>  0.6740
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.3306
#>  1.1924
#>  0.6999
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.0031
#> -0.5660
#>  0.0235
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2436
#> -2.2715
#> -1.3631
#> -0.3225
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.8512
#>  0.0334
#>  1.0734
#>  2.2791
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.2775
#>  2.0649
#>  0.7591
#> -0.0965
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.4159
#>  0.4118
#> -0.7226
#>  1.0895
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.8616
#>  0.9163
#>  1.0176
#> -0.8928
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  0.5088
#> -0.1886
#> -0.9851
#> -1.1011
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.0726
#> -0.8237
#> -0.3192
#> -0.7509
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.6198
#> -1.2828
#> -0.3089
#>  0.2668
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.9968
#>  1.7800
#> -0.0981
#>  0.6959
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.1826
#>  0.2707
#>  0.2966
#> -0.5139
#> [ CPUFloatType{4} ]
#> 
#> 
```
