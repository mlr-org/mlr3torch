# Target Batchgetter for a Task

Returns the function that converts the target column(s) of a `task` into
the target tensor `y` of a batch, i.e. the tensor that the loss is
applied to. The returned function takes an argument `data`, a
[`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html)
containing only the target column(s), and returns a
[`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html).
It is `NULL` for a task with no target at all, whose batches have no `y`
element and whose loss is called as `loss(y_hat)`.

For the target encodings of the built-in task types, see section
*Network Head and Target Encoding* of
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).

When adding support for a custom task type, implement a method for the
corresponding [`Task`](https://mlr3.mlr-org.com/reference/Task.html)
class.

## Usage

``` r
get_target_batchgetter(task, ...)
```

## Arguments

- task:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  The task.

- ...:

  (any)  
  Additional arguments. Not used yet.

## Value

`function(data)`, `function(data, x)`, or `NULL` for a task with no
target

## Examples

``` r
batchgetter = get_target_batchgetter(tsk("iris"))
batchgetter(data.table::data.table(Species = factor(c("setosa", "virginica"))))
#> torch_tensor
#>  1
#>  2
#> [ CPULongType{2} ]
```
