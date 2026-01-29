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
#>  0.3963  0.9654 -0.3141
#>  0.0198  0.9101  1.7965
#> -1.4149  1.2543 -0.3160
#>  0.7922 -0.5850 -0.7214
#> -0.8265  0.7039 -0.6114
#> -2.1631 -1.8974  0.9619
#>  0.3207 -0.3886 -0.1243
#> -0.5500 -2.0905 -0.4180
#>  1.1283  1.2954 -0.9348
#>  0.7684 -0.1014  0.4073
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.3963
#>  0.9654
#> -0.3141
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  0.0198
#>  0.9101
#>  1.7965
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#> -1.4149
#>  1.2543
#> -0.3160
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#>  0.7922
#> -0.5850
#> -0.7214
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.8265
#>  0.7039
#> -0.6114
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -2.1631
#> -1.8974
#>  0.9619
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.3207
#> -0.3886
#> -0.1243
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#> -0.5500
#> -2.0905
#> -0.4180
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  1.1283
#>  1.2954
#> -0.9348
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.7684
#> -0.1014
#>  0.4073
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.3963  0.9654 -0.3141
#>  0.0198  0.9101  1.7965
#> -1.4149  1.2543 -0.3160
#>  0.7922 -0.5850 -0.7214
#> -0.8265  0.7039 -0.6114
#> -2.1631 -1.8974  0.9619
#>  0.3207 -0.3886 -0.1243
#> -0.5500 -2.0905 -0.4180
#>  1.1283  1.2954 -0.9348
#>  0.7684 -0.1014  0.4073
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  0.3033  1.4501  0.2781  0.7800
#>  1.0002  0.9460  0.2188 -0.2927
#>  0.2651  0.2683 -0.4306 -0.8050
#>  1.1216  0.7723 -0.9168 -0.3751
#>  0.8487 -1.5061  0.4241 -1.4962
#> -0.1219  0.2722  0.0746 -0.8537
#>  0.1131 -0.4164  0.1621 -0.6573
#>  0.7188 -0.9696 -0.1927  1.2184
#> -0.8694 -1.0846  0.1332 -0.2722
#>  0.4903  0.8187 -1.3955  0.6614
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.3963
#>  0.9654
#> -0.3141
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#>  0.0198
#>  0.9101
#>  1.7965
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#> -1.4149
#>  1.2543
#> -0.3160
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#>  0.7922
#> -0.5850
#> -0.7214
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.8265
#>  0.7039
#> -0.6114
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -2.1631
#> -1.8974
#>  0.9619
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.3207
#> -0.3886
#> -0.1243
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#> -0.5500
#> -2.0905
#> -0.4180
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  1.1283
#>  1.2954
#> -0.9348
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.7684
#> -0.1014
#>  0.4073
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  0.3033
#>  1.4501
#>  0.2781
#>  0.7800
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#>  1.0002
#>  0.9460
#>  0.2188
#> -0.2927
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.2651
#>  0.2683
#> -0.4306
#> -0.8050
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.1216
#>  0.7723
#> -0.9168
#> -0.3751
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  0.8487
#> -1.5061
#>  0.4241
#> -1.4962
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.1219
#>  0.2722
#>  0.0746
#> -0.8537
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  0.1131
#> -0.4164
#>  0.1621
#> -0.6573
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.7188
#> -0.9696
#> -0.1927
#>  1.2184
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#> -0.8694
#> -1.0846
#>  0.1332
#> -0.2722
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#>  0.4903
#>  0.8187
#> -1.3955
#>  0.6614
#> [ CPUFloatType{4} ]
#> 
#> 
```
