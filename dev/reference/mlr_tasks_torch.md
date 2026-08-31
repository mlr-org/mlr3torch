# Generic Torch Task

A general-purpose [`Task`](https://mlr3.mlr-org.com/reference/Task.html)
that can be used for modeling arbitrary problems, including supervised
and unsupervised problems. The article on *Custom Learning Problems*
covers all of this in more detail.

The problem this generic task solves is that it is rather complicated to
register new task types with `mlr3`, so this class makes this easier.
The price of this flexibility is the loss of some compatibility checks.

## Super class

[`mlr3::Task`](https://mlr3.mlr-org.com/reference/Task.html) -\>
`TaskTorch`

## Active bindings

- `hash`:

  (`character(1)`)  
  The hash of the task.

- `default_encoder`:

  (`function()` or `NULL`)  
  The default prediction encoder. Read-only.

- `default_measure`:

  ([`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) or
  `NULL`)  
  See the construction argument. Read-only, for the same reason as
  `default_encoder`.

- `output_dim`:

  (`function()` or `NULL`)  
  See the construction argument. Called by
  [`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md).

## Methods

### Public methods

- [`TaskTorch$new()`](#method-TaskTorch-initialize)

- [`TaskTorch$truth()`](#method-TaskTorch-truth)

- [`TaskTorch$clone()`](#method-TaskTorch-clone)

Inherited methods

- [`mlr3::Task$add_strata()`](https://mlr3.mlr-org.com/reference/Task.html#method-add_strata)
- [`mlr3::Task$cbind()`](https://mlr3.mlr-org.com/reference/Task.html#method-cbind)
- [`mlr3::Task$data()`](https://mlr3.mlr-org.com/reference/Task.html#method-data)
- [`mlr3::Task$divide()`](https://mlr3.mlr-org.com/reference/Task.html#method-divide)
- [`mlr3::Task$droplevels()`](https://mlr3.mlr-org.com/reference/Task.html#method-droplevels)
- [`mlr3::Task$filter()`](https://mlr3.mlr-org.com/reference/Task.html#method-filter)
- [`mlr3::Task$format()`](https://mlr3.mlr-org.com/reference/Task.html#method-format)
- [`mlr3::Task$formula()`](https://mlr3.mlr-org.com/reference/Task.html#method-formula)
- [`mlr3::Task$head()`](https://mlr3.mlr-org.com/reference/Task.html#method-head)
- [`mlr3::Task$help()`](https://mlr3.mlr-org.com/reference/Task.html#method-help)
- [`mlr3::Task$levels()`](https://mlr3.mlr-org.com/reference/Task.html#method-levels)
- [`mlr3::Task$materialize_view()`](https://mlr3.mlr-org.com/reference/Task.html#method-materialize_view)
- [`mlr3::Task$missings()`](https://mlr3.mlr-org.com/reference/Task.html#method-missings)
- [`mlr3::Task$print()`](https://mlr3.mlr-org.com/reference/Task.html#method-print)
- [`mlr3::Task$rbind()`](https://mlr3.mlr-org.com/reference/Task.html#method-rbind)
- [`mlr3::Task$rename()`](https://mlr3.mlr-org.com/reference/Task.html#method-rename)
- [`mlr3::Task$select()`](https://mlr3.mlr-org.com/reference/Task.html#method-select)
- [`mlr3::Task$set_col_roles()`](https://mlr3.mlr-org.com/reference/Task.html#method-set_col_roles)
- [`mlr3::Task$set_levels()`](https://mlr3.mlr-org.com/reference/Task.html#method-set_levels)
- [`mlr3::Task$set_row_roles()`](https://mlr3.mlr-org.com/reference/Task.html#method-set_row_roles)

------------------------------------------------------------------------

### `TaskTorch$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    TaskTorch$new(
      id,
      backend,
      target = NULL,
      label = NA_character_,
      output_dim = NULL,
      default_encoder = NULL,
      default_measure = NULL
    )

#### Arguments

- `id`:

  (`character(1)`)  
  The id of the task.

- `backend`:

  ([`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html)
  or [`data.frame()`](https://rdrr.io/r/base/data.frame.html))  
  The data.

- `target`:

  ([`character()`](https://rdrr.io/r/base/character.html) or `NULL`)  
  The names of the target columns. `NULL` (default) for a task without a
  target, see section *Tasks without a Target* of `TaskTorch`.

- `label`:

  (`character(1)`)  
  The label of the task.

- `output_dim`:

  (`function()` or `NULL`)  
  Returns the number of output units the network needs. Takes an
  argument `task` and returns a single positive integer. May be `NULL`
  (default), in which case any caller of
  [`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md)
  errors.

- `default_encoder`:

  (`function()` or `NULL`)  
  The default prediction encoder for the task. This can be overwritten
  by a learner's private `$.encode_prediction` method. See
  [`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
  for more information.

- `default_measure`:

  ([`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) or
  `NULL`)  
  The default measure of the task, i.e. what
  [`msr("torch.default")`](https://mlr3torch.mlr-org.com/dev/reference/mlr_measures_torch.default.md)
  resolves to.

------------------------------------------------------------------------

### `TaskTorch$truth()`

The ground truth, see section *Scoring*. Might return `NULL` for
unsupervised problems.

#### Usage

    TaskTorch$truth(rows = NULL)

#### Arguments

- `rows`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The rows to return the truth for. All rows if `NULL`.

------------------------------------------------------------------------

### `TaskTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    TaskTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# multi-label classification: one logical column per label
d = data.frame(x1 = rnorm(50), x2 = rnorm(50))
d$a = d$x1 > 0
d$b = d$x2 > 0
task = as_task_torch(d, target = c("a", "b"), id = "labels",
  output_dim = function(task) length(task$target_names),
  default_encoder = function(task, network_output, predict_type) {
    prob = as.matrix(torch::nnf_sigmoid(network_output)$cpu())
    colnames(prob) = task$target_names
    list(response = prob > 0.5, prob = if (predict_type == "prob") prob)
  })
task
#> 
#> ── <TaskTorch> (50x4) ──────────────────────────────────────────────────────────
#> • Target: a and b
#> • Properties: -
#> • Features (2):
#>   • dbl (2): x1, x2
output_dim_for(task)
```
