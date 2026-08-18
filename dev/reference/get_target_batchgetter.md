# Target Batchgetter for a Task

Returns the function that converts the target column(s) of a `task` into
the target tensor `y` of a batch.

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

`function(data)`

## Examples

``` r
task = tsk("iris")
batchgetter = get_target_batchgetter(task)
batchgetter(task$data(1:2, "Species")[[1L]])
#> torch_tensor
#>  1
#> [ CPULongType{1} ]
```
