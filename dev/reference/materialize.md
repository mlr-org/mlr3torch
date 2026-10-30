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
#> -0.4279  1.2651  1.3791
#>  0.6316  1.2427 -0.0577
#> -0.7768 -1.0138 -0.7052
#> -1.2883 -0.1644 -1.5877
#>  1.3795  0.2221 -0.5224
#> -0.1327  2.2189 -0.3262
#> -0.9363 -2.5653 -1.6665
#> -1.4764  0.1158  1.9392
#> -2.2924 -0.8828  0.0701
#> -0.2810 -0.3733  0.6043
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.4279
#>  1.2651
#>  1.3791
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.6316
#>  1.2427
#> -0.0577
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.7768
#> -1.0138
#> -0.7052
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -1.2883
#> -0.1644
#> -1.5877
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  1.3795
#>  0.2221
#> -0.5224
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.1327
#>  2.2189
#> -0.3262
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.9363
#> -2.5653
#> -1.6665
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.4764
#>  0.1158
#>  1.9392
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -2.2924
#> -0.8828
#>  0.0701
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.2810
#> -0.3733
#>  0.6043
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.4279  1.2651  1.3791
#>  0.6316  1.2427 -0.0577
#> -0.7768 -1.0138 -0.7052
#> -1.2883 -0.1644 -1.5877
#>  1.3795  0.2221 -0.5224
#> -0.1327  2.2189 -0.3262
#> -0.9363 -2.5653 -1.6665
#> -1.4764  0.1158  1.9392
#> -2.2924 -0.8828  0.0701
#> -0.2810 -0.3733  0.6043
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.2084 -0.7119  0.5412 -1.3292
#> -0.1054  1.4613  0.6820 -1.4592
#>  0.5890  0.5158  0.7508  0.4168
#> -0.6201  0.2609 -0.4024  0.7302
#>  0.0031 -2.2696 -0.2191 -0.9738
#> -0.6878 -0.3459  0.1881  0.2434
#> -1.7946  1.1093 -1.3071  2.1793
#> -0.3559  0.6055  0.1569  0.4422
#> -0.4020 -1.2772 -0.5562  0.0490
#>  1.6356  1.3614 -1.9832  0.5276
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.4279
#>  1.2651
#>  1.3791
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.6316
#>  1.2427
#> -0.0577
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.7768
#> -1.0138
#> -0.7052
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -1.2883
#> -0.1644
#> -1.5877
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  1.3795
#>  0.2221
#> -0.5224
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.1327
#>  2.2189
#> -0.3262
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.9363
#> -2.5653
#> -1.6665
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.4764
#>  0.1158
#>  1.9392
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -2.2924
#> -0.8828
#>  0.0701
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.2810
#> -0.3733
#>  0.6043
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.2084
#> -0.7119
#>  0.5412
#> -1.3292
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.1054
#>  1.4613
#>  0.6820
#> -1.4592
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5890
#>  0.5158
#>  0.7508
#>  0.4168
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.6201
#>  0.2609
#> -0.4024
#>  0.7302
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.0031
#> -2.2696
#> -0.2191
#> -0.9738
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.6878
#> -0.3459
#>  0.1881
#>  0.2434
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -1.7946
#>  1.1093
#> -1.3071
#>  2.1793
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.3559
#>  0.6055
#>  0.1569
#>  0.4422
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.4020
#> -1.2772
#> -0.5562
#>  0.0490
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.6356
#>  1.3614
#> -1.9832
#>  0.5276
#> [ CPUFloatType{4} ]
#> 
#> 
```
