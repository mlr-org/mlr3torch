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
#> -0.3059 -0.2231  1.1030
#> -1.9255  1.1472  0.6580
#>  0.0458  0.4887  0.2849
#> -0.9476 -0.1891  0.1879
#> -0.4573 -0.2395  1.9047
#>  0.0015 -0.1296  0.0391
#>  1.3153  1.7860 -0.7857
#>  1.0341 -0.1005  0.0058
#> -1.0309  0.4584  0.0640
#>  1.0110 -0.4517 -0.6661
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.3059
#> -0.2231
#>  1.1030
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.9255
#>  1.1472
#>  0.6580
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.0458
#>  0.4887
#>  0.2849
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.9476
#> -0.1891
#>  0.1879
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.4573
#> -0.2395
#>  1.9047
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  0.0015
#> -0.1296
#>  0.0391
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.3153
#>  1.7860
#> -0.7857
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.0341
#> -0.1005
#>  0.0058
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -1.0309
#>  0.4584
#>  0.0640
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.0110
#> -0.4517
#> -0.6661
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.3059 -0.2231  1.1030
#> -1.9255  1.1472  0.6580
#>  0.0458  0.4887  0.2849
#> -0.9476 -0.1891  0.1879
#> -0.4573 -0.2395  1.9047
#>  0.0015 -0.1296  0.0391
#>  1.3153  1.7860 -0.7857
#>  1.0341 -0.1005  0.0058
#> -1.0309  0.4584  0.0640
#>  1.0110 -0.4517 -0.6661
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.4660 -0.3894 -1.5402  0.0454
#>  0.1341  0.9324 -1.0932  0.0935
#>  0.6629  0.4123  1.7252  0.2302
#>  0.2272 -0.9183 -1.3440  0.2584
#> -1.0589 -0.0018  1.3045 -0.1486
#> -0.7453  0.3911  0.6376  1.5768
#>  0.2325  1.4752  0.5200 -1.2329
#>  0.8147  0.0894 -1.5101  0.0496
#>  0.7728 -0.9437 -0.2519 -1.6525
#>  0.8153  0.5735 -0.8569  0.4409
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.3059
#> -0.2231
#>  1.1030
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.9255
#>  1.1472
#>  0.6580
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.0458
#>  0.4887
#>  0.2849
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.9476
#> -0.1891
#>  0.1879
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.4573
#> -0.2395
#>  1.9047
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  0.0015
#> -0.1296
#>  0.0391
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.3153
#>  1.7860
#> -0.7857
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.0341
#> -0.1005
#>  0.0058
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -1.0309
#>  0.4584
#>  0.0640
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.0110
#> -0.4517
#> -0.6661
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.4660
#> -0.3894
#> -1.5402
#>  0.0454
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.1341
#>  0.9324
#> -1.0932
#>  0.0935
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.6629
#>  0.4123
#>  1.7252
#>  0.2302
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.2272
#> -0.9183
#> -1.3440
#>  0.2584
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.0589
#> -0.0018
#>  1.3045
#> -0.1486
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.7453
#>  0.3911
#>  0.6376
#>  1.5768
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.2325
#>  1.4752
#>  0.5200
#> -1.2329
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.8147
#>  0.0894
#> -1.5101
#>  0.0496
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.7728
#> -0.9437
#> -0.2519
#> -1.6525
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.8153
#>  0.5735
#> -0.8569
#>  0.4409
#> [ CPUFloatType{4} ]
#> 
#> 
```
