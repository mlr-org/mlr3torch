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
#> -2.4268 -1.1175 -0.7627
#>  0.6673  0.2780  0.2274
#>  1.4977  0.7147 -0.4520
#> -1.4626  0.1264  1.8858
#> -1.0419 -0.1525 -1.7573
#>  0.1446  0.2607 -0.8492
#>  1.0007 -1.0609  0.7682
#>  0.1814  0.0498  1.3558
#>  0.4376  0.2461 -0.9651
#>  0.8530 -0.1636  0.1297
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -2.4268
#> -1.1175
#> -0.7627
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.6673
#>  0.2780
#>  0.2274
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.4977
#>  0.7147
#> -0.4520
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.4626
#>  0.1264
#>  1.8858
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -1.0419
#> -0.1525
#> -1.7573
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.1446
#>  0.2607
#> -0.8492
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.0007
#> -1.0609
#>  0.7682
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.1814
#>  0.0498
#>  1.3558
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.4376
#>  0.2461
#> -0.9651
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.8530
#> -0.1636
#>  0.1297
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -2.4268 -1.1175 -0.7627
#>  0.6673  0.2780  0.2274
#>  1.4977  0.7147 -0.4520
#> -1.4626  0.1264  1.8858
#> -1.0419 -0.1525 -1.7573
#>  0.1446  0.2607 -0.8492
#>  1.0007 -1.0609  0.7682
#>  0.1814  0.0498  1.3558
#>  0.4376  0.2461 -0.9651
#>  0.8530 -0.1636  0.1297
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2471  0.9985 -0.0645 -0.1238
#>  0.4557  0.8436  0.8947  0.5939
#> -0.7677  1.3911  0.8111  0.6020
#> -0.3615  1.0239 -0.1830  0.1945
#> -1.0419  2.0482 -0.5354 -0.1977
#>  1.1102  0.2692  0.1348  0.7178
#>  0.6729 -0.8683 -1.6058  0.1806
#>  0.2469 -1.5777 -0.4395  0.7837
#>  1.5054  0.3052  0.3237 -0.3332
#> -0.8340 -0.7848  0.0876 -1.4028
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -2.4268
#> -1.1175
#> -0.7627
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.6673
#>  0.2780
#>  0.2274
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.4977
#>  0.7147
#> -0.4520
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.4626
#>  0.1264
#>  1.8858
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -1.0419
#> -0.1525
#> -1.7573
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.1446
#>  0.2607
#> -0.8492
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.0007
#> -1.0609
#>  0.7682
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.1814
#>  0.0498
#>  1.3558
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.4376
#>  0.2461
#> -0.9651
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.8530
#> -0.1636
#>  0.1297
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2471
#>  0.9985
#> -0.0645
#> -0.1238
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.4557
#>  0.8436
#>  0.8947
#>  0.5939
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.7677
#>  1.3911
#>  0.8111
#>  0.6020
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.3615
#>  1.0239
#> -0.1830
#>  0.1945
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.0419
#>  2.0482
#> -0.5354
#> -0.1977
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.1102
#>  0.2692
#>  0.1348
#>  0.7178
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.6729
#> -0.8683
#> -1.6058
#>  0.1806
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.2469
#> -1.5777
#> -0.4395
#>  0.7837
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.5054
#>  0.3052
#>  0.3237
#> -0.3332
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.8340
#> -0.7848
#>  0.0876
#> -1.4028
#> [ CPUFloatType{4} ]
#> 
#> 
```
