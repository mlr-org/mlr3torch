# Materialize Lazy Tensor Columns

This will materialize a
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
or a [`data.frame()`](https://rdrr.io/r/base/data.frame.html) /
[`list()`](https://rdrr.io/r/base/list.html) containing – among other
things –
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
columns. I.e. the data described in the underlying
[`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)s
is loaded for the indices in the
[`lazy_tensor()`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md),
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
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
  or a [`list()`](https://rdrr.io/r/base/list.html) /
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html) containing
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
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

  (`character(1)` or
  [`environment()`](https://rdrr.io/r/base/environment.html) or
  `NULL`)  
  Optional cache for (intermediate) materialization results. Per
  default, caching will be enabled when the same dataset or data
  descriptor (with different output pointer) is used for more than one
  lazy tensor column.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)s
or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md))

## Details

Materializing a lazy tensor consists of:

1.  Loading the data from the internal dataset of the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md).

2.  Processing these batches in the preprocessing
    [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s.

3.  Returning the result of the
    [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
    pointed to by the
    [`DataDescriptor`](https://mlr3torch.mlr-org.com/reference/DataDescriptor.md)
    (`pointer`).

With multiple
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
columns we can benefit from caching because: a) Output(s) from the
dataset might be input to multiple graphs. b) Different lazy tensors
might be outputs from the same graph.

For this reason it is possible to provide a cache environment. The hash
key for a) is the hash of the indices and the dataset. The hash key for
b) is the hash of the indices, dataset and preprocessing graph.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#>  0.3833  0.0565 -0.7408
#>  1.2860 -0.5447 -1.4973
#> -0.9837  0.4664  1.1152
#> -0.8656  0.6834 -0.7507
#>  0.1531  0.6582 -0.1727
#> -0.9355  0.2941 -1.4062
#> -0.5507 -1.5277 -0.4647
#> -1.0829  0.1786 -0.0003
#>  0.6536  1.7219 -0.1497
#>  0.5895 -0.2417  1.5167
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3833
#>  0.0565
#> -0.7408
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.2860
#> -0.5447
#> -1.4973
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.9837
#>  0.4664
#>  1.1152
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.8656
#>  0.6834
#> -0.7507
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.1531
#>  0.6582
#> -0.1727
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.9355
#>  0.2941
#> -1.4062
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.5507
#> -1.5277
#> -0.4647
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -1.0829
#>  0.1786
#> -0.0003
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.6536
#>  1.7219
#> -0.1497
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.5895
#> -0.2417
#>  1.5167
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3833  0.0565 -0.7408
#>  1.2860 -0.5447 -1.4973
#> -0.9837  0.4664  1.1152
#> -0.8656  0.6834 -0.7507
#>  0.1531  0.6582 -0.1727
#> -0.9355  0.2941 -1.4062
#> -0.5507 -1.5277 -0.4647
#> -1.0829  0.1786 -0.0003
#>  0.6536  1.7219 -0.1497
#>  0.5895 -0.2417  1.5167
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -0.8219  1.2121  0.5599  0.0679
#> -0.8585 -0.3130  1.7675  0.0590
#> -0.1433  0.6107 -0.5994  1.7670
#> -0.6237  0.9522 -0.8463 -0.8843
#> -0.4721  0.4179 -0.3604 -1.2839
#> -0.1028  2.0977 -1.0083  2.1550
#> -0.5994 -0.5404  0.4277 -0.7149
#>  0.8480  0.3316  0.0991 -0.1677
#> -0.2750  1.1852 -1.7449 -0.1954
#>  1.1873  0.8449  0.0372  1.3506
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3833
#>  0.0565
#> -0.7408
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.2860
#> -0.5447
#> -1.4973
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.9837
#>  0.4664
#>  1.1152
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.8656
#>  0.6834
#> -0.7507
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.1531
#>  0.6582
#> -0.1727
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.9355
#>  0.2941
#> -1.4062
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.5507
#> -1.5277
#> -0.4647
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -1.0829
#>  0.1786
#> -0.0003
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.6536
#>  1.7219
#> -0.1497
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.5895
#> -0.2417
#>  1.5167
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -0.8219
#>  1.2121
#>  0.5599
#>  0.0679
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.8585
#> -0.3130
#>  1.7675
#>  0.0590
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.1433
#>  0.6107
#> -0.5994
#>  1.7670
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.6237
#>  0.9522
#> -0.8463
#> -0.8843
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.4721
#>  0.4179
#> -0.3604
#> -1.2839
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1028
#>  2.0977
#> -1.0083
#>  2.1550
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.5994
#> -0.5404
#>  0.4277
#> -0.7149
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.8480
#>  0.3316
#>  0.0991
#> -0.1677
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.2750
#>  1.1852
#> -1.7449
#> -0.1954
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  1.1873
#>  0.8449
#>  0.0372
#>  1.3506
#> [ CPUFloatType{4} ]
#> 
#> 
```
