# Create a Generic Torch Task

Creates a
[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md),
the general-purpose task type of `mlr3torch`, from a
[`data.frame()`](https://rdrr.io/r/base/data.frame.html) or a
[`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html).
See the *Custom Learning Problems* article for more information.

## Usage

``` r
as_task_torch(x, target = NULL, id = deparse(substitute(x))[1L], ...)
```

## Arguments

- x:

  ([`data.frame()`](https://rdrr.io/r/base/data.frame.html) or
  [`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html))  
  The data.

- target:

  ([`character()`](https://rdrr.io/r/base/character.html) or `NULL`)  
  The names of the target columns. `NULL` (default) for a task without a
  target, see section *Tasks without a Target* of
  [`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md).

- id:

  (`character(1)`)  
  The id of the task.

- ...:

  (any)  
  Further arguments passed to
  [`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md)`$new()`,
  such as `output_dim`, `default_encoder` or `default_measure`.

## Value

[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md)

## Examples

``` r
# multi-output regression
d = data.frame(x = rnorm(50))
d$y1 = d$x + rnorm(50)
d$y2 = 2 * d$x + rnorm(50)
as_task_torch(d, target = c("y1", "y2"))
#> 
#> ── <TaskTorch> (50x3) ──────────────────────────────────────────────────────────
#> • Target: y1 and y2
#> • Properties: -
#> • Features (1):
#>   • dbl (1): x

# unsupervised: no target at all
as_task_torch(data.frame(a = rnorm(50), b = rnorm(50)))
#> 
#> ── <TaskTorch> (50x2) ──────────────────────────────────────────────────────────
#> • Target:
#> • Properties: -
#> • Features (2):
#>   • dbl (2): a, b
```
