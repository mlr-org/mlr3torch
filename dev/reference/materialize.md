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
#>  1.1878 -0.1307  0.2696
#>  0.8209 -0.0349  0.6470
#>  0.3557 -1.2101  0.0623
#> -0.3163 -0.5623  1.9617
#>  0.3390 -2.2947 -1.1720
#>  1.5739  0.3769  1.7360
#>  0.1928 -1.3048  0.7740
#>  0.0502 -0.0824  0.6250
#> -0.4376 -0.3271 -0.4398
#>  0.8831  0.4513  0.4203
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.1878
#> -0.1307
#>  0.2696
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.8209
#> -0.0349
#>  0.6470
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.3557
#> -1.2101
#>  0.0623
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.3163
#> -0.5623
#>  1.9617
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.3390
#> -2.2947
#> -1.1720
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.5739
#>  0.3769
#>  1.7360
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.1928
#> -1.3048
#>  0.7740
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.0502
#> -0.0824
#>  0.6250
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.4376
#> -0.3271
#> -0.4398
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.8831
#>  0.4513
#>  0.4203
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.1878 -0.1307  0.2696
#>  0.8209 -0.0349  0.6470
#>  0.3557 -1.2101  0.0623
#> -0.3163 -0.5623  1.9617
#>  0.3390 -2.2947 -1.1720
#>  1.5739  0.3769  1.7360
#>  0.1928 -1.3048  0.7740
#>  0.0502 -0.0824  0.6250
#> -0.4376 -0.3271 -0.4398
#>  0.8831  0.4513  0.4203
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.6659 -0.1612 -0.6069  0.9742
#> -0.3193 -0.6563  1.0733 -0.1178
#> -1.8510 -2.0299 -1.1637 -1.1103
#> -0.3036  0.0082  0.8604  0.1970
#> -0.1393  0.3611  0.0843 -0.0454
#>  1.1650 -1.6267  1.6525 -0.8632
#> -0.4187  0.6935  1.3410 -0.2787
#> -2.2434 -0.1618  1.4285  0.4409
#> -2.2401  0.4679 -0.5554 -0.1812
#> -0.8088 -0.8182 -0.5802 -0.1722
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.1878
#> -0.1307
#>  0.2696
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.8209
#> -0.0349
#>  0.6470
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.3557
#> -1.2101
#>  0.0623
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.3163
#> -0.5623
#>  1.9617
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.3390
#> -2.2947
#> -1.1720
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.5739
#>  0.3769
#>  1.7360
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.1928
#> -1.3048
#>  0.7740
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.0502
#> -0.0824
#>  0.6250
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.4376
#> -0.3271
#> -0.4398
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.8831
#>  0.4513
#>  0.4203
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.6659
#> -0.1612
#> -0.6069
#>  0.9742
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.3193
#> -0.6563
#>  1.0733
#> -0.1178
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.8510
#> -2.0299
#> -1.1637
#> -1.1103
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.3036
#>  0.0082
#>  0.8604
#>  0.1970
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.1393
#>  0.3611
#>  0.0843
#> -0.0454
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.1650
#> -1.6267
#>  1.6525
#> -0.8632
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.4187
#>  0.6935
#>  1.3410
#> -0.2787
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -2.2434
#> -0.1618
#>  1.4285
#>  0.4409
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -2.2401
#>  0.4679
#> -0.5554
#> -0.1812
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.8088
#> -0.8182
#> -0.5802
#> -0.1722
#> [ CPUFloatType{4} ]
#> 
#> 
```
