# Batch Constructor for a Task

Returns the function that builds a whole batch of a `task`, i.e. both
the features `x` and the target `y`. This is how
[`task_dataset`](https://mlr3torch.mlr-org.com/dev/reference/task_dataset.md)
loads its batches.

The returned function takes the arguments

- `data`, a
  [`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html)
  with the feature *and* target columns of the batch, and

- `cache`, a rather internal
  [`hashtab`](https://rdrr.io/r/utils/hashtab.html) that can be used as
  a cache when loading multiple
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  columns.

It returns a [`list()`](https://rdrr.io/r/base/list.html) with elements
`x` (a named [`list()`](https://rdrr.io/r/base/list.html) of
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)s)
and `y` (a
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
or `NULL`).

The default method applies the ingress tokens to obtain `x` and the
target batchgetter to obtain `y`.

## Usage

``` r
get_batch_constructor(
  task,
  feature_ingress_tokens,
  target_batchgetter = NULL,
  ...
)
```

## Arguments

- task:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  The task.

- feature_ingress_tokens:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchIngressToken`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md))  
  The ingress tokens that define `x`. Their features are already
  resolved, i.e. they are
  [`character()`](https://rdrr.io/r/base/character.html) vectors and not
  [`Selector`](https://mlr3pipelines.mlr-org.com/reference/Selector.html)s.

- target_batchgetter:

  (`function()` or `NULL`)  
  Converts the target columns of a batch into the target tensor `y` that
  the loss is applied to. Takes an argument `data`, a
  [`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html)
  with only the target columns, and optionally an argument `x`, the
  named list of feature tensors of the batch, which is what a target
  that is a function of the input needs, see
  [`get_target_batchgetter()`](https://mlr3torch.mlr-org.com/dev/reference/get_target_batchgetter.md).
  If `NULL` (default), the batches have no `y` element.

- ...:

  (any)  
  Additional arguments. Not used yet.

## Value

`function(data, cache) -> list(x = list<torch_tensor>, y = torch_tensor | NULL)`

## Examples

``` r
task = tsk("iris")
token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4))
# the token's features are a `Selector`; `task_dataset()` resolves them for you
token$features = token$features(task)
batch_constructor = get_batch_constructor(
  task,
  feature_ingress_tokens = list(input = token),
  target_batchgetter = get_target_batchgetter(task)
)
batch = batch_constructor(data = task$data(rows = 1:2))
batch$x$input
#> torch_tensor
#>  1.4000  0.2000  5.1000  3.5000
#>  1.4000  0.2000  4.9000  3.0000
#> [ CPUFloatType{2,4} ]
batch$y
#> torch_tensor
#>  1
#>  1
#> [ CPULongType{2} ]
```
