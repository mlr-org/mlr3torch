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
#> -1.8200 -0.7895 -2.1304
#> -0.0623 -0.0002  1.3260
#> -0.5945  1.2502 -0.5310
#> -0.1921 -0.5244 -0.6220
#>  0.4001 -0.2705  0.1286
#> -1.4425  1.3145 -0.8837
#>  0.6184 -0.8661  0.0874
#>  1.6787  1.1090  0.3251
#>  1.9394 -0.1853 -0.4974
#> -0.2497 -1.5104 -0.3861
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -1.8200
#> -0.7895
#> -2.1304
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.0623
#> -0.0002
#>  1.3260
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -0.5945
#>  1.2502
#> -0.5310
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.1921
#> -0.5244
#> -0.6220
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#>  0.4001
#> -0.2705
#>  0.1286
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -1.4425
#>  1.3145
#> -0.8837
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.6184
#> -0.8661
#>  0.0874
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  1.6787
#>  1.1090
#>  0.3251
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.9394
#> -0.1853
#> -0.4974
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -0.2497
#> -1.5104
#> -0.3861
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -1.8200 -0.7895 -2.1304
#> -0.0623 -0.0002  1.3260
#> -0.5945  1.2502 -0.5310
#> -0.1921 -0.5244 -0.6220
#>  0.4001 -0.2705  0.1286
#> -1.4425  1.3145 -0.8837
#>  0.6184 -0.8661  0.0874
#>  1.6787  1.1090  0.3251
#>  1.9394 -0.1853 -0.4974
#> -0.2497 -1.5104 -0.3861
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.3585  0.1097  0.7483 -1.2078
#>  0.9801 -1.5864  1.7321  0.6393
#> -0.4146  1.2942  0.2506 -1.2306
#> -0.4290  1.2962  0.4582 -0.9476
#>  0.0335  0.4760 -1.5830  1.2111
#> -0.7064  0.2058  0.4506 -0.2132
#>  0.3795  0.2002  0.3127 -0.6593
#> -0.0642  0.2933  0.2321 -0.0251
#>  0.3991  0.5416 -0.8492 -1.2706
#> -0.9241 -0.2734  0.0402 -1.2263
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -1.8200
#> -0.7895
#> -2.1304
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.0623
#> -0.0002
#>  1.3260
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -0.5945
#>  1.2502
#> -0.5310
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.1921
#> -0.5244
#> -0.6220
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#>  0.4001
#> -0.2705
#>  0.1286
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -1.4425
#>  1.3145
#> -0.8837
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.6184
#> -0.8661
#>  0.0874
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  1.6787
#>  1.1090
#>  0.3251
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.9394
#> -0.1853
#> -0.4974
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -0.2497
#> -1.5104
#> -0.3861
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.3585
#>  0.1097
#>  0.7483
#> -1.2078
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  0.9801
#> -1.5864
#>  1.7321
#>  0.6393
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.4146
#>  1.2942
#>  0.2506
#> -1.2306
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#> -0.4290
#>  1.2962
#>  0.4582
#> -0.9476
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.0335
#>  0.4760
#> -1.5830
#>  1.2111
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.7064
#>  0.2058
#>  0.4506
#> -0.2132
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.3795
#>  0.2002
#>  0.3127
#> -0.6593
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.0642
#>  0.2933
#>  0.2321
#> -0.0251
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.3991
#>  0.5416
#> -0.8492
#> -1.2706
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.9241
#> -0.2734
#>  0.0402
#> -1.2263
#> [ CPUFloatType{4} ]
#> 
#> 
```
