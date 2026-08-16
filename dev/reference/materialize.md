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
#>  0.6706 -0.6449 -1.0899
#>  0.5073 -0.7457  0.2930
#> -0.7621  0.8003 -0.2081
#> -0.7912  0.0008 -1.5396
#>  0.5826  2.0506 -0.0469
#>  0.2074  0.3080  2.2132
#> -0.2235  0.1896  1.2560
#>  0.0806  0.0990 -1.5092
#>  0.0177 -0.8251  1.4496
#>  1.0293  0.8361 -1.0813
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.6706
#> -0.6449
#> -1.0899
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.5073
#> -0.7457
#>  0.2930
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.7621
#>  0.8003
#> -0.2081
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.7912
#>  0.0008
#> -1.5396
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.5826
#>  2.0506
#> -0.0469
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.2074
#>  0.3080
#>  2.2132
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.2235
#>  0.1896
#>  1.2560
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.0806
#>  0.0990
#> -1.5092
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.0177
#> -0.8251
#>  1.4496
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.0293
#>  0.8361
#> -1.0813
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.6706 -0.6449 -1.0899
#>  0.5073 -0.7457  0.2930
#> -0.7621  0.8003 -0.2081
#> -0.7912  0.0008 -1.5396
#>  0.5826  2.0506 -0.0469
#>  0.2074  0.3080  2.2132
#> -0.2235  0.1896  1.2560
#>  0.0806  0.0990 -1.5092
#>  0.0177 -0.8251  1.4496
#>  1.0293  0.8361 -1.0813
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2151 -1.1371 -0.8470 -0.9002
#>  0.2158  1.2042 -0.4487 -0.3178
#>  0.2654  0.3054 -0.3116  0.8645
#> -0.2884 -0.2866  0.7029 -0.9460
#> -0.1262 -0.9146  2.4314 -0.2951
#> -0.6425  0.3728  0.2225 -0.8410
#> -0.5084 -0.5913  0.6405  0.7696
#>  0.4307 -0.5037 -0.2557 -1.9463
#> -0.8664  0.6819 -0.1516  0.0353
#>  0.1360 -0.0247 -0.0616 -0.1379
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.6706
#> -0.6449
#> -1.0899
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.5073
#> -0.7457
#>  0.2930
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.7621
#>  0.8003
#> -0.2081
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.7912
#>  0.0008
#> -1.5396
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.5826
#>  2.0506
#> -0.0469
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.2074
#>  0.3080
#>  2.2132
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.2235
#>  0.1896
#>  1.2560
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.0806
#>  0.0990
#> -1.5092
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.0177
#> -0.8251
#>  1.4496
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.0293
#>  0.8361
#> -1.0813
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2151
#> -1.1371
#> -0.8470
#> -0.9002
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.2158
#>  1.2042
#> -0.4487
#> -0.3178
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.2654
#>  0.3054
#> -0.3116
#>  0.8645
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.2884
#> -0.2866
#>  0.7029
#> -0.9460
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.1262
#> -0.9146
#>  2.4314
#> -0.2951
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.6425
#>  0.3728
#>  0.2225
#> -0.8410
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.5084
#> -0.5913
#>  0.6405
#>  0.7696
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.4307
#> -0.5037
#> -0.2557
#> -1.9463
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.8664
#>  0.6819
#> -0.1516
#>  0.0353
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.1360
#> -0.0247
#> -0.0616
#> -0.1379
#> [ CPUFloatType{4} ]
#> 
#> 
```
