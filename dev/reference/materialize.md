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

For this reason it is possible to provide a cache environment. The hash
key for a) is the hash of the indices and the dataset. The hash key for
b) is the hash of the indices, dataset and preprocessing graph.

## Examples

``` r
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
#> torch_tensor
#>  0.6616 -1.2092  0.3113
#> -0.3344 -0.6531  0.0576
#>  0.0587 -1.1405  1.5523
#> -0.5867  0.9528 -1.7471
#> -0.3158  1.4079  0.0602
#>  1.0230 -0.5660 -0.6662
#>  0.2349  2.3392 -0.3709
#>  0.6756  0.8452 -0.3174
#> -0.9063 -0.7318 -0.2507
#>  0.7335 -0.6884  1.1614
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#>  0.6616
#> -1.2092
#>  0.3113
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.3344
#> -0.6531
#>  0.0576
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.0587
#> -1.1405
#>  1.5523
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.5867
#>  0.9528
#> -1.7471
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.3158
#>  1.4079
#>  0.0602
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#>  1.0230
#> -0.5660
#> -0.6662
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  0.2349
#>  2.3392
#> -0.3709
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.6756
#>  0.8452
#> -0.3174
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#> -0.9063
#> -0.7318
#> -0.2507
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#>  0.7335
#> -0.6884
#>  1.1614
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#>  0.6616 -1.2092  0.3113
#> -0.3344 -0.6531  0.0576
#>  0.0587 -1.1405  1.5523
#> -0.5867  0.9528 -1.7471
#> -0.3158  1.4079  0.0602
#>  1.0230 -0.5660 -0.6662
#>  0.2349  2.3392 -0.3709
#>  0.6756  0.8452 -0.3174
#> -0.9063 -0.7318 -0.2507
#>  0.7335 -0.6884  1.1614
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#> -2.2840  0.5105  0.1254  0.8047
#> -1.4333 -0.6383  0.4430  0.7250
#>  0.8166 -0.5199 -0.8227 -0.0712
#>  0.7565  0.2666 -1.6437 -1.6717
#>  1.2561  0.3188  1.0680 -0.7886
#>  1.2913 -1.2919  1.0426 -1.4532
#>  1.2813  0.3135  0.3921  0.8692
#> -1.3950  0.3835  0.6045 -0.4758
#>  0.7747  2.5041 -0.5435  0.6630
#> -1.9322  1.2033 -0.2343  1.0170
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#>  0.6616
#> -1.2092
#>  0.3113
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -0.3344
#> -0.6531
#>  0.0576
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.0587
#> -1.1405
#>  1.5523
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.5867
#>  0.9528
#> -1.7471
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.3158
#>  1.4079
#>  0.0602
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#>  1.0230
#> -0.5660
#> -0.6662
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  0.2349
#>  2.3392
#> -0.3709
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.6756
#>  0.8452
#> -0.3174
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#> -0.9063
#> -0.7318
#> -0.2507
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#>  0.7335
#> -0.6884
#>  1.1614
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#> -2.2840
#>  0.5105
#>  0.1254
#>  0.8047
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -1.4333
#> -0.6383
#>  0.4430
#>  0.7250
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.8166
#> -0.5199
#> -0.8227
#> -0.0712
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  0.7565
#>  0.2666
#> -1.6437
#> -1.6717
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#>  1.2561
#>  0.3188
#>  1.0680
#> -0.7886
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#>  1.2913
#> -1.2919
#>  1.0426
#> -1.4532
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#>  1.2813
#>  0.3135
#>  0.3921
#>  0.8692
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#> -1.3950
#>  0.3835
#>  0.6045
#> -0.4758
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.7747
#>  2.5041
#> -0.5435
#>  0.6630
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -1.9322
#>  1.2033
#> -0.2343
#>  1.0170
#> [ CPUFloatType{4} ]
#> 
#> 
```
