# Materialize a Lazy Tensor

Convert a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
to a
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html).

## Usage

``` r
materialize_internal(x, device = "cpu", cache = NULL, rbind)
```

## Arguments

- x:

  ([`lazy_tensor()`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md))  
  The lazy tensor to materialize.

- device:

  (`character(1L)`)  
  The device to put the materialized tensor on (after running the
  preprocessing graph).

- cache:

  (`NULL` or
  [`environment()`](https://rdrr.io/r/base/environment.html))  
  Whether to cache the (intermediate) results of the materialization.
  This can make data loading faster when multiple `lazy_tensor`s
  reference the same dataset or graph.

- rbind:

  (`logical(1)`)  
  Whtether to rbind the resulting tensors (`TRUE`) or return them as a
  list of tensors (`FALSE`).

## Value

[`lazy_tensor()`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)

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

When materializing multiple
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
columns, caching can be useful because: a) Output(s) from the dataset
might be input to multiple graphs. (in task_dataset this is shoudl
rarely be the case because because we try to merge them). b) Different
lazy tensors might be outputs from the same graph.

For this reason it is possible to provide a cache environment. The hash
key for a) is the hash of the indices and the dataset. The hash key for
b) is the hash of the indices dataset and preprocessing graph.
