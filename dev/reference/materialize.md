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
#>  0.0377  1.9300  0.3184
#> -1.0500 -0.2351 -0.0698
#>  0.8298  0.3100  1.6291
#>  1.1153 -1.3143 -0.0450
#> -0.2611 -0.5521 -0.3588
#> -1.8737 -0.3401 -0.6295
#>  0.1746  0.3026 -0.9852
#>  0.2807 -0.8147 -0.3476
#>  0.6906  0.5782  0.8053
#> -0.1580  1.5130  0.4897
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.0377
#>  1.9300
#>  0.3184
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.0500
#> -0.2351
#> -0.0698
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.8298
#>  0.3100
#>  1.6291
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  1.1153
#> -1.3143
#> -0.0450
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.2611
#> -0.5521
#> -0.3588
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -1.8737
#> -0.3401
#> -0.6295
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.1746
#>  0.3026
#> -0.9852
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.2807
#> -0.8147
#> -0.3476
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.6906
#>  0.5782
#>  0.8053
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.1580
#>  1.5130
#>  0.4897
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.0377  1.9300  0.3184
#> -1.0500 -0.2351 -0.0698
#>  0.8298  0.3100  1.6291
#>  1.1153 -1.3143 -0.0450
#> -0.2611 -0.5521 -0.3588
#> -1.8737 -0.3401 -0.6295
#>  0.1746  0.3026 -0.9852
#>  0.2807 -0.8147 -0.3476
#>  0.6906  0.5782  0.8053
#> -0.1580  1.5130  0.4897
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.2093 -0.2001 -0.6840 -1.8518
#>  0.0264 -0.5690  1.1097 -0.2428
#> -1.6105 -0.9461 -0.4669  1.0980
#>  1.4087 -2.7110 -1.0932 -1.0994
#> -0.9743 -2.3793 -0.0093  0.8931
#> -1.5261 -0.3241 -0.8978  0.6110
#>  1.0534 -0.2935  0.0049 -2.0311
#>  1.0471 -1.3611  1.5775  0.8369
#>  0.3592  1.4620 -0.1056 -1.7076
#>  0.8877 -0.5997 -0.2296 -1.4547
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.0377
#>  1.9300
#>  0.3184
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.0500
#> -0.2351
#> -0.0698
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.8298
#>  0.3100
#>  1.6291
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  1.1153
#> -1.3143
#> -0.0450
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.2611
#> -0.5521
#> -0.3588
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -1.8737
#> -0.3401
#> -0.6295
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.1746
#>  0.3026
#> -0.9852
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.2807
#> -0.8147
#> -0.3476
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.6906
#>  0.5782
#>  0.8053
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.1580
#>  1.5130
#>  0.4897
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.2093
#> -0.2001
#> -0.6840
#> -1.8518
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.0264
#> -0.5690
#>  1.1097
#> -0.2428
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -1.6105
#> -0.9461
#> -0.4669
#>  1.0980
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.4087
#> -2.7110
#> -1.0932
#> -1.0994
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.9743
#> -2.3793
#> -0.0093
#>  0.8931
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -1.5261
#> -0.3241
#> -0.8978
#>  0.6110
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  1.0534
#> -0.2935
#>  0.0049
#> -2.0311
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.0471
#> -1.3611
#>  1.5775
#>  0.8369
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.3592
#>  1.4620
#> -0.1056
#> -1.7076
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.8877
#> -0.5997
#> -0.2296
#> -1.4547
#> [ CPUFloatType{4} ]
#> 
#> 
```
