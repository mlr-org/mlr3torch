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

  (`character(1)` or
  [`environment()`](https://rdrr.io/r/base/environment.html) or
  `NULL`)  
  Optional cache for (intermediate) materialization results. Per
  default, caching will be enabled when the same dataset or data
  descriptor (with different output pointer) is used for more than one
  lazy tensor column.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)s
or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md))

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

For this reason it is possible to provide a cache environment. The hash
key for a) is the hash of the indices and the dataset. The hash key for
b) is the hash of the indices, dataset and preprocessing graph.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#>  1.3042 -0.4038  0.6035
#>  1.6807  1.0177  0.5118
#>  1.0964 -1.5854 -0.6598
#>  1.8442 -0.8431 -0.9858
#> -0.1681  0.1472 -0.2264
#> -0.4325  1.7581 -0.5108
#> -0.3274  0.0045 -0.5396
#> -0.5989  0.9620 -0.2225
#>  0.0494  0.7550  2.4691
#>  1.1687 -0.0316  0.5336
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  1.3042
#> -0.4038
#>  0.6035
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1.6807
#>  1.0177
#>  0.5118
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  1.0964
#> -1.5854
#> -0.6598
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  1.8442
#> -0.8431
#> -0.9858
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.1681
#>  0.1472
#> -0.2264
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -0.4325
#>  1.7581
#> -0.5108
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#> -0.3274
#>  0.0045
#> -0.5396
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.5989
#>  0.9620
#> -0.2225
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.0494
#>  0.7550
#>  2.4691
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  1.1687
#> -0.0316
#>  0.5336
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  1.3042 -0.4038  0.6035
#>  1.6807  1.0177  0.5118
#>  1.0964 -1.5854 -0.6598
#>  1.8442 -0.8431 -0.9858
#> -0.1681  0.1472 -0.2264
#> -0.4325  1.7581 -0.5108
#> -0.3274  0.0045 -0.5396
#> -0.5989  0.9620 -0.2225
#>  0.0494  0.7550  2.4691
#>  1.1687 -0.0316  0.5336
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.1054 -1.0758 -1.1677 -1.5450
#> -0.8334 -0.2332 -0.6523 -0.6947
#> -0.3642 -0.9340 -1.3727  0.5829
#>  0.1339 -1.9106 -0.5405 -2.1086
#> -0.9563 -0.4795  0.0543 -0.0174
#>  1.6434 -0.5543  0.7244  2.3419
#>  0.7080  0.7110  0.4471 -1.9796
#> -0.3344  0.5042  0.2983 -1.8557
#>  1.8242 -0.0009 -1.0956  0.4010
#> -2.0213 -0.7897 -0.7870 -0.8939
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  1.3042
#> -0.4038
#>  0.6035
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  1.6807
#>  1.0177
#>  0.5118
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  1.0964
#> -1.5854
#> -0.6598
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  1.8442
#> -0.8431
#> -0.9858
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.1681
#>  0.1472
#> -0.2264
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -0.4325
#>  1.7581
#> -0.5108
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#> -0.3274
#>  0.0045
#> -0.5396
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.5989
#>  0.9620
#> -0.2225
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.0494
#>  0.7550
#>  2.4691
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  1.1687
#> -0.0316
#>  0.5336
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.1054
#> -1.0758
#> -1.1677
#> -1.5450
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.8334
#> -0.2332
#> -0.6523
#> -0.6947
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#> -0.3642
#> -0.9340
#> -1.3727
#>  0.5829
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.1339
#> -1.9106
#> -0.5405
#> -2.1086
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -0.9563
#> -0.4795
#>  0.0543
#> -0.0174
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.6434
#> -0.5543
#>  0.7244
#>  2.3419
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.7080
#>  0.7110
#>  0.4471
#> -1.9796
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -0.3344
#>  0.5042
#>  0.2983
#> -1.8557
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  1.8242
#> -0.0009
#> -1.0956
#>  0.4010
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -2.0213
#> -0.7897
#> -0.7870
#> -0.8939
#> [ CPUFloatType{4} ]
#> 
#> 
```
