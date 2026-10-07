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
#>  0.7594  1.1250  0.5090
#> -2.1906 -1.0579 -0.8799
#>  2.1035 -0.0404  0.0772
#> -0.9835 -1.7047 -1.0122
#> -0.4703 -0.4418  1.0774
#>  0.4185 -1.4668 -0.8766
#>  0.5080 -0.4051  0.1622
#> -1.3687  0.2155  0.3176
#> -1.0374  1.3124  2.2706
#>  1.7429 -1.1380  1.9628
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.7594
#>  1.1250
#>  0.5090
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -2.1906
#> -1.0579
#> -0.8799
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  2.1035
#> -0.0404
#>  0.0772
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.9835
#> -1.7047
#> -1.0122
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.4703
#> -0.4418
#>  1.0774
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.4185
#> -1.4668
#> -0.8766
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.5080
#> -0.4051
#>  0.1622
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.3687
#>  0.2155
#>  0.3176
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.0374
#>  1.3124
#>  2.2706
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.7429
#> -1.1380
#>  1.9628
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.7594  1.1250  0.5090
#> -2.1906 -1.0579 -0.8799
#>  2.1035 -0.0404  0.0772
#> -0.9835 -1.7047 -1.0122
#> -0.4703 -0.4418  1.0774
#>  0.4185 -1.4668 -0.8766
#>  0.5080 -0.4051  0.1622
#> -1.3687  0.2155  0.3176
#> -1.0374  1.3124  2.2706
#>  1.7429 -1.1380  1.9628
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -1.1259 -0.8843 -0.4749 -0.4490
#>  0.4657  0.6034  0.6777 -0.1035
#> -3.0449 -0.0547 -0.2497 -0.2423
#> -0.2634  1.0890  1.2980  1.1263
#>  0.7950  0.6277  0.0864  0.6839
#> -0.7922 -0.2242 -0.2812 -0.3751
#> -0.0275 -0.5866 -1.5203  1.4542
#> -1.6486 -0.4794 -0.6661  0.7671
#>  0.0761  0.6273  1.5215  0.7544
#> -0.2404 -1.1260 -0.5153  0.3503
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.7594
#>  1.1250
#>  0.5090
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -2.1906
#> -1.0579
#> -0.8799
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  2.1035
#> -0.0404
#>  0.0772
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.9835
#> -1.7047
#> -1.0122
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.4703
#> -0.4418
#>  1.0774
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.4185
#> -1.4668
#> -0.8766
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.5080
#> -0.4051
#>  0.1622
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.3687
#>  0.2155
#>  0.3176
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.0374
#>  1.3124
#>  2.2706
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.7429
#> -1.1380
#>  1.9628
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -1.1259
#> -0.8843
#> -0.4749
#> -0.4490
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.4657
#>  0.6034
#>  0.6777
#> -0.1035
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -3.0449
#> -0.0547
#> -0.2497
#> -0.2423
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.2634
#>  1.0890
#>  1.2980
#>  1.1263
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.7950
#>  0.6277
#>  0.0864
#>  0.6839
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.7922
#> -0.2242
#> -0.2812
#> -0.3751
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.0275
#> -0.5866
#> -1.5203
#>  1.4542
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.6486
#> -0.4794
#> -0.6661
#>  0.7671
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.0761
#>  0.6273
#>  1.5215
#>  0.7544
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.2404
#> -1.1260
#> -0.5153
#>  0.3503
#> [ CPUFloatType{4} ]
#> 
#> 
```
