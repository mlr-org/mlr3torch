# Measure for a Generic Torch Task

Wraps a plain R function into a
[`Measure`](https://mlr3.mlr-org.com/reference/Measure.html) that scores
the predictions of a
[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md).
Use
[`msr_torch()`](https://mlr3torch.mlr-org.com/dev/reference/msr_torch.md)
to construct one. See the *Custom Learning Problems* article for how to
create and use such measures.

## See also

Other Measure:
[`mlr_measures_torch.default`](https://mlr3torch.mlr-org.com/dev/reference/mlr_measures_torch.default.md)

## Super class

[`mlr3::Measure`](https://mlr3.mlr-org.com/reference/Measure.html) -\>
`MeasureTorch`

## Active bindings

- `hash`:

  (`character(1)`)  
  The hash of the measure.

## Methods

### Public methods

- [`MeasureTorch$new()`](#method-MeasureTorch-initialize)

- [`MeasureTorch$clone()`](#method-MeasureTorch-clone)

Inherited methods

- [`mlr3::Measure$aggregate()`](https://mlr3.mlr-org.com/reference/Measure.html#method-aggregate)
- [`mlr3::Measure$format()`](https://mlr3.mlr-org.com/reference/Measure.html#method-format)
- [`mlr3::Measure$help()`](https://mlr3.mlr-org.com/reference/Measure.html#method-help)
- [`mlr3::Measure$obs_loss()`](https://mlr3.mlr-org.com/reference/Measure.html#method-obs_loss)
- [`mlr3::Measure$print()`](https://mlr3.mlr-org.com/reference/Measure.html#method-print)
- [`mlr3::Measure$score()`](https://mlr3.mlr-org.com/reference/Measure.html#method-score)

------------------------------------------------------------------------

### `MeasureTorch$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    MeasureTorch$new(
      id,
      fun,
      minimize = NA,
      range = c(-Inf, Inf),
      predict_type = "response",
      properties = character(),
      label = NA_character_,
      obs_loss = NULL
    )

#### Arguments

- `id`:

  (`character(1)`)  
  The id of the measure.

- `fun`:

  (`function()`)  
  The scoring function. It receives whichever of the arguments `truth`,
  `response`, `prob`, `se`, `lazy_tensor`, `prediction`, `task`,
  `learner`, `train_set` and `weights` it declares, and must return a
  single number. Asking for anything else is an error. An argument that
  the prediction does not have – `weights` on a task without a
  `weights_measure` column, or `prob` on a response-only prediction – is
  not passed at all, so a default declared for it is what the function
  sees. `weights` are also withheld when the measure's `$use_weights` is
  set to `"ignore"`, so give the argument a default if the measure
  should score with and without weights, and none if it cannot do
  without them – asking for weights that do not exist is an error, not a
  score.

- `minimize`:

  (`logical(1)`)  
  Whether a smaller score is better. `NA` (default) means the direction
  is unknown.

- `range`:

  (`numeric(2)`)  
  The range of possible scores.

- `predict_type`:

  (`character(1)`)  
  The predict type the measure requires: `"response"` (default),
  `"prob"`, `"se"` or `"lazy_tensor"`. A measure asking for one it did
  not declare here still receives it, if the prediction has it.

------------------------------------------------------------------------

### `MeasureTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    MeasureTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
d = data.frame(x = rnorm(10), y = rnorm(10))
task = as_task_torch(d, target = "y")
measure = msr_torch("mse", function(truth, response) mean((truth - response)^2))
measure
#> 
#> ── <MeasureTorch> (mse) ────────────────────────────────────────────────────────
#> • Packages: mlr3
#> • Range: [-Inf, Inf]
#> • Minimize: NA
#> • Average: macro
#> • Parameters: list()
#> • Properties: -
#> • Predict type: response
#> • Predict sets: test
#> • Aggregator: mean()
```
