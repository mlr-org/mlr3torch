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
#>  0.3033  0.1651  1.3996
#>  1.2373 -0.1058 -0.4066
#>  0.6811 -3.4718  0.9569
#> -0.2473 -1.7870 -0.5552
#>  0.4154 -0.9259 -0.1463
#>  0.7145 -0.0835 -0.4376
#> -0.3146 -2.0971  0.8307
#>  0.4377  0.4576 -0.0583
#>  0.4361 -0.5668  0.6291
#> -0.1786  2.1176  0.4332
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3033
#>  0.1651
#>  1.3996
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.2373
#> -0.1058
#> -0.4066
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.6811
#> -3.4718
#>  0.9569
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.2473
#> -1.7870
#> -0.5552
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.4154
#> -0.9259
#> -0.1463
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.7145
#> -0.0835
#> -0.4376
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.3146
#> -2.0971
#>  0.8307
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.4377
#>  0.4576
#> -0.0583
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.4361
#> -0.5668
#>  0.6291
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.1786
#>  2.1176
#>  0.4332
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3033  0.1651  1.3996
#>  1.2373 -0.1058 -0.4066
#>  0.6811 -3.4718  0.9569
#> -0.2473 -1.7870 -0.5552
#>  0.4154 -0.9259 -0.1463
#>  0.7145 -0.0835 -0.4376
#> -0.3146 -2.0971  0.8307
#>  0.4377  0.4576 -0.0583
#>  0.4361 -0.5668  0.6291
#> -0.1786  2.1176  0.4332
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.5235  0.9566  0.1934  0.3974
#> -0.7406  0.3289  0.0765 -1.5322
#>  1.6973 -0.4769  0.3713 -0.8866
#>  1.4331 -0.0763 -0.9102  1.9837
#>  1.0607 -0.4637  2.4228 -0.5867
#> -0.5945  0.8308  0.3245 -1.8409
#>  0.5224 -0.2651 -1.1929  2.0675
#>  0.2046  0.4909  0.6889 -0.1859
#>  1.7639  0.6031  0.1372 -1.1675
#>  0.4480  0.3482  0.2693 -0.4525
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3033
#>  0.1651
#>  1.3996
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.2373
#> -0.1058
#> -0.4066
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.6811
#> -3.4718
#>  0.9569
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.2473
#> -1.7870
#> -0.5552
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.4154
#> -0.9259
#> -0.1463
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.7145
#> -0.0835
#> -0.4376
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.3146
#> -2.0971
#>  0.8307
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.4377
#>  0.4576
#> -0.0583
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.4361
#> -0.5668
#>  0.6291
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.1786
#>  2.1176
#>  0.4332
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.5235
#>  0.9566
#>  0.1934
#>  0.3974
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.7406
#>  0.3289
#>  0.0765
#> -1.5322
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.6973
#> -0.4769
#>  0.3713
#> -0.8866
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.4331
#> -0.0763
#> -0.9102
#>  1.9837
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.0607
#> -0.4637
#>  2.4228
#> -0.5867
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.5945
#>  0.8308
#>  0.3245
#> -1.8409
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.5224
#> -0.2651
#> -1.1929
#>  2.0675
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.2046
#>  0.4909
#>  0.6889
#> -0.1859
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.7639
#>  0.6031
#>  0.1372
#> -1.1675
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.4480
#>  0.3482
#>  0.2693
#> -0.4525
#> [ CPUFloatType{4} ]
#> 
#> 
```
