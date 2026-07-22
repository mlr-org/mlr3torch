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
#> -0.3031  1.7202 -0.0647
#> -1.3657  0.0169  0.0546
#>  0.3682 -0.7427 -1.0141
#> -0.4226 -1.9780 -0.6071
#> -0.6874 -0.5678 -0.2922
#> -2.4208  0.7655 -1.1926
#>  1.5063  0.1970  0.7499
#>  0.1918  0.4620  0.3837
#>  0.8359 -1.3885 -0.2703
#> -1.1094  1.2918 -1.0307
#> [ CPUFloatType{10,3} ]
materialize(lt1, rbind = FALSE)
#> [[1]]
#> torch_tensor
#> -0.3031
#>  1.7202
#> -0.0647
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.3657
#>  0.0169
#>  0.0546
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.3682
#> -0.7427
#> -1.0141
#> [ CPUFloatType{3} ]
#> 
#> [[4]]
#> torch_tensor
#> -0.4226
#> -1.9780
#> -0.6071
#> [ CPUFloatType{3} ]
#> 
#> [[5]]
#> torch_tensor
#> -0.6874
#> -0.5678
#> -0.2922
#> [ CPUFloatType{3} ]
#> 
#> [[6]]
#> torch_tensor
#> -2.4208
#>  0.7655
#> -1.1926
#> [ CPUFloatType{3} ]
#> 
#> [[7]]
#> torch_tensor
#>  1.5063
#>  0.1970
#>  0.7499
#> [ CPUFloatType{3} ]
#> 
#> [[8]]
#> torch_tensor
#>  0.1918
#>  0.4620
#>  0.3837
#> [ CPUFloatType{3} ]
#> 
#> [[9]]
#> torch_tensor
#>  0.8359
#> -1.3885
#> -0.2703
#> [ CPUFloatType{3} ]
#> 
#> [[10]]
#> torch_tensor
#> -1.1094
#>  1.2918
#> -1.0307
#> [ CPUFloatType{3} ]
#> 
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
#> $lt1
#> torch_tensor
#> -0.3031  1.7202 -0.0647
#> -1.3657  0.0169  0.0546
#>  0.3682 -0.7427 -1.0141
#> -0.4226 -1.9780 -0.6071
#> -0.6874 -0.5678 -0.2922
#> -2.4208  0.7655 -1.1926
#>  1.5063  0.1970  0.7499
#>  0.1918  0.4620  0.3837
#>  0.8359 -1.3885 -0.2703
#> -1.1094  1.2918 -1.0307
#> [ CPUFloatType{10,3} ]
#> 
#> $lt2
#> torch_tensor
#>  1.4700 -1.8191 -0.0961  0.4818
#> -0.2632  0.0761  0.1876 -0.6690
#>  0.5694  0.5195 -0.6072  1.5828
#>  1.0362 -0.8807  1.4912 -0.6683
#> -1.1433 -0.9262 -0.0641 -3.5822
#> -0.5160  0.5686  0.0279  1.4885
#> -0.0675  1.0611 -0.3730  0.2766
#>  0.5060 -1.6831 -1.1624 -0.9570
#>  0.4583  0.0736 -0.0633 -1.0330
#> -0.0873 -1.1907 -0.2939  0.6065
#> [ CPUFloatType{10,4} ]
#> 
materialize(d, rbind = FALSE)
#> $lt1
#> $lt1[[1]]
#> torch_tensor
#> -0.3031
#>  1.7202
#> -0.0647
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[2]]
#> torch_tensor
#> -1.3657
#>  0.0169
#>  0.0546
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[3]]
#> torch_tensor
#>  0.3682
#> -0.7427
#> -1.0141
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[4]]
#> torch_tensor
#> -0.4226
#> -1.9780
#> -0.6071
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[5]]
#> torch_tensor
#> -0.6874
#> -0.5678
#> -0.2922
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[6]]
#> torch_tensor
#> -2.4208
#>  0.7655
#> -1.1926
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[7]]
#> torch_tensor
#>  1.5063
#>  0.1970
#>  0.7499
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[8]]
#> torch_tensor
#>  0.1918
#>  0.4620
#>  0.3837
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[9]]
#> torch_tensor
#>  0.8359
#> -1.3885
#> -0.2703
#> [ CPUFloatType{3} ]
#> 
#> $lt1[[10]]
#> torch_tensor
#> -1.1094
#>  1.2918
#> -1.0307
#> [ CPUFloatType{3} ]
#> 
#> 
#> $lt2
#> $lt2[[1]]
#> torch_tensor
#>  1.4700
#> -1.8191
#> -0.0961
#>  0.4818
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[2]]
#> torch_tensor
#> -0.2632
#>  0.0761
#>  0.1876
#> -0.6690
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[3]]
#> torch_tensor
#>  0.5694
#>  0.5195
#> -0.6072
#>  1.5828
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[4]]
#> torch_tensor
#>  1.0362
#> -0.8807
#>  1.4912
#> -0.6683
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[5]]
#> torch_tensor
#> -1.1433
#> -0.9262
#> -0.0641
#> -3.5822
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[6]]
#> torch_tensor
#> -0.5160
#>  0.5686
#>  0.0279
#>  1.4885
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[7]]
#> torch_tensor
#> -0.0675
#>  1.0611
#> -0.3730
#>  0.2766
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[8]]
#> torch_tensor
#>  0.5060
#> -1.6831
#> -1.1624
#> -0.9570
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[9]]
#> torch_tensor
#>  0.4583
#>  0.0736
#> -0.0633
#> -1.0330
#> [ CPUFloatType{4} ]
#> 
#> $lt2[[10]]
#> torch_tensor
#> -0.0873
#> -1.1907
#> -0.2939
#>  0.6065
#> [ CPUFloatType{4} ]
#> 
#> 
```
