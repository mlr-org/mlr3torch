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
#> -0.2444  0.2329 -1.0453
#> -1.9100  0.2817  1.6587
#>  0.4920 -0.4197  1.1691
#>  2.3842  0.0663  0.7118
#> -0.7566  0.0938 -0.9300
#>  1.4082  1.5414 -0.5836
#>  2.5962 -0.0084 -0.1184
#> -0.2315 -0.6272 -0.7189
#>  0.2558  1.4725 -0.4896
#> -0.0729  2.0779  1.0536
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.2444
#>  0.2329
#> -1.0453
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.9100
#>  0.2817
#>  1.6587
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.4920
#> -0.4197
#>  1.1691
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  2.3842
#>  0.0663
#>  0.7118
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.7566
#>  0.0938
#> -0.9300
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.4082
#>  1.5414
#> -0.5836
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  2.5962
#> -0.0084
#> -0.1184
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.2315
#> -0.6272
#> -0.7189
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.2558
#>  1.4725
#> -0.4896
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.0729
#>  2.0779
#>  1.0536
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.2444  0.2329 -1.0453
#> -1.9100  0.2817  1.6587
#>  0.4920 -0.4197  1.1691
#>  2.3842  0.0663  0.7118
#> -0.7566  0.0938 -0.9300
#>  1.4082  1.5414 -0.5836
#>  2.5962 -0.0084 -0.1184
#> -0.2315 -0.6272 -0.7189
#>  0.2558  1.4725 -0.4896
#> -0.0729  2.0779  1.0536
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.5386  1.0710  0.4166  0.7659
#>  1.0414 -0.7869 -0.9461  0.8244
#>  1.4461 -1.4240  0.8008  1.5133
#>  0.2701  0.2840  0.6701  3.1026
#> -0.2816 -0.7887  0.7570 -0.7310
#> -0.9470 -0.5760  1.9467  1.2171
#> -0.3700 -0.3164  0.2583 -0.3521
#>  1.0553  0.1132 -0.4656 -0.1098
#> -0.3897 -0.5110 -0.2358 -0.5155
#> -1.5003  0.2299 -1.7207  0.4415
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.2444
#>  0.2329
#> -1.0453
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.9100
#>  0.2817
#>  1.6587
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.4920
#> -0.4197
#>  1.1691
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  2.3842
#>  0.0663
#>  0.7118
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.7566
#>  0.0938
#> -0.9300
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.4082
#>  1.5414
#> -0.5836
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  2.5962
#> -0.0084
#> -0.1184
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.2315
#> -0.6272
#> -0.7189
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.2558
#>  1.4725
#> -0.4896
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.0729
#>  2.0779
#>  1.0536
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.5386
#>  1.0710
#>  0.4166
#>  0.7659
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.0414
#> -0.7869
#> -0.9461
#>  0.8244
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  1.4461
#> -1.4240
#>  0.8008
#>  1.5133
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2701
#>  0.2840
#>  0.6701
#>  3.1026
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.2816
#> -0.7887
#>  0.7570
#> -0.7310
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.9470
#> -0.5760
#>  1.9467
#>  1.2171
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.3700
#> -0.3164
#>  0.2583
#> -0.3521
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  1.0553
#>  0.1132
#> -0.4656
#> -0.1098
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.3897
#> -0.5110
#> -0.2358
#> -0.5155
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.5003
#>  0.2299
#> -1.7207
#>  0.4415
#> [ CPUFloatType{4} ]
#> 
#> 
```
