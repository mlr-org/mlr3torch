# Prediction Object for a Generic Torch Task

The [`Prediction`](https://mlr3.mlr-org.com/reference/Prediction.html)
object returned by learners that were trained on a
[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md).

Because a
[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md)
can represent very different learning problems, this class does not
prescribe much about how `truth`, `response`, `prob` and `se` are
stored: the task's prediction encoder decides, and that encoder is
yours. The one rule is that the *first* dimension indexes the
observations. Within that, an element may be an atomic vector, a
[`matrix()`](https://rdrr.io/r/base/matrix.html), an
[`array()`](https://rdrr.io/r/base/array.html) of any dimensionality, a
[`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html) or a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md).
`truth` is whatever `task$truth()` returned – a vector for one target, a
`data.table` for several, a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
for a lazy tensor column, and nothing at all for a task without a
target.

When a prediction is converted to a `data.table`, which is e.g. used for
the printer, the conversion depends on the type of the object. A `prob`
*matrix* spreads into one column per class, the way it does for a
classification prediction. Everything else that is wider than one value
per observation becomes a single column whose cells hold that
observation's own array, printed as its shape – `<array[3]>` for a
response matrix with three columns, `<array[3x224x224]>` for a `prob`
with a class dimension and two spatial ones.

## Missing Predictions

Only a response with a single value per observation can report a
*missing* prediction, so `$missing` can only be `TRUE` for scalars and
is always `FALSE` for parially missing predictions.

## Super class

[`mlr3::Prediction`](https://mlr3.mlr-org.com/reference/Prediction.html)
-\> `PredictionTorch`

## Active bindings

- `response`:

  (any)  
  The predicted response.

- `prob`:

  (any)  
  The predicted probabilities.

- `se`:

  (any)  
  The standard errors of the prediction.

- `lazy_tensor`:

  ([`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  or [`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html)
  of them)  
  The output of the network, for the predict type `"lazy_tensor"`.

## Methods

### Public methods

- [`PredictionTorch$new()`](#method-PredictionTorch-initialize)

- [`PredictionTorch$clone()`](#method-PredictionTorch-clone)

Inherited methods

- [`mlr3::Prediction$filter()`](https://mlr3.mlr-org.com/reference/Prediction.html#method-filter)
- [`mlr3::Prediction$format()`](https://mlr3.mlr-org.com/reference/Prediction.html#method-format)
- [`mlr3::Prediction$help()`](https://mlr3.mlr-org.com/reference/Prediction.html#method-help)
- [`mlr3::Prediction$obs_loss()`](https://mlr3.mlr-org.com/reference/Prediction.html#method-obs_loss)
- [`mlr3::Prediction$print()`](https://mlr3.mlr-org.com/reference/Prediction.html#method-print)
- [`mlr3::Prediction$score()`](https://mlr3.mlr-org.com/reference/Prediction.html#method-score)

------------------------------------------------------------------------

### `PredictionTorch$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PredictionTorch$new(
      task = NULL,
      row_ids = task$row_ids,
      truth = if (!is.null(task)) task$truth(row_ids),
      response = NULL,
      prob = NULL,
      se = NULL,
      lazy_tensor = NULL,
      weights = NULL,
      check = TRUE
    )

#### Arguments

- `task`:

  ([`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md))  
  The task that was predicted on. Used to extract the default `row_ids`
  and the `truth`.

- `row_ids`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The row ids of the predicted observations.

- `truth`:

  (any)  
  The ground truth, i.e. what `task$truth()` returned.

- `response`:

  (any)  
  The predicted response.

- `prob`:

  (any)  
  The predicted probabilities.

- `se`:

  (any)  
  The standard errors of the prediction.

- `lazy_tensor`:

  ([`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  or [`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html)
  of them)  
  The output of the network, see the predict type `"lazy_tensor"` of
  [`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).
  A network with more than one head produces one column per head.

- `weights`:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html) or `NULL`)  
  The measure weights of the predicted observations, i.e. the
  `weights_measure` column of the task. `mlr3` fills this in, so it
  rarely has to be passed by hand.

- `check`:

  (`logical(1)`)  
  Whether to check the consistency of the prediction data.

------------------------------------------------------------------------

### `PredictionTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PredictionTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
d = data.frame(x = rnorm(10), y1 = rnorm(10), y2 = rnorm(10))
task = as_task_torch(d, target = c("y1", "y2"))
PredictionTorch$new(task, response = as.matrix(d[, c("y1", "y2")]))
#> 
#> ── <PredictionTorch> for 10 observations: ──────────────────────────────────────
#>  row_ids    truth.y1    truth.y2   response
#>        1 -0.55369938  0.46815442 <array[2]>
#>        2  0.62898204  0.36295126 <array[2]>
#>        3  2.06502490 -1.30454355 <array[2]>
#>      ---         ---         ---        ---
#>        8 -0.05260191 -0.01595031 <array[2]>
#>        9  0.54299634 -0.82678895 <array[2]>
#>       10 -0.91407483 -1.51239965 <array[2]>
```
