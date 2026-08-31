# Default Measure of a Generic Torch Task

This is a simple placeholder measure and extracts the actual value from
the `$default_measure` of a
[`TaskTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_tasks_torch.md).

## See also

Other Measure:
[`mlr_measures_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_measures_torch.md)

## Super class

[`mlr3::Measure`](https://mlr3.mlr-org.com/reference/Measure.html) -\>
`MeasureTorchDefault`

## Methods

### Public methods

- [`MeasureTorchDefault$new()`](#method-MeasureTorchDefault-initialize)

- [`MeasureTorchDefault$clone()`](#method-MeasureTorchDefault-clone)

Inherited methods

- [`mlr3::Measure$aggregate()`](https://mlr3.mlr-org.com/reference/Measure.html#method-aggregate)
- [`mlr3::Measure$format()`](https://mlr3.mlr-org.com/reference/Measure.html#method-format)
- [`mlr3::Measure$help()`](https://mlr3.mlr-org.com/reference/Measure.html#method-help)
- [`mlr3::Measure$obs_loss()`](https://mlr3.mlr-org.com/reference/Measure.html#method-obs_loss)
- [`mlr3::Measure$print()`](https://mlr3.mlr-org.com/reference/Measure.html#method-print)
- [`mlr3::Measure$score()`](https://mlr3.mlr-org.com/reference/Measure.html#method-score)

------------------------------------------------------------------------

### `MeasureTorchDefault$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    MeasureTorchDefault$new()

------------------------------------------------------------------------

### `MeasureTorchDefault$clone()`

The objects of this class are cloneable with this method.

#### Usage

    MeasureTorchDefault$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
d = data.frame(x = rnorm(10), y = rnorm(10))
task = as_task_torch(d, target = "y",
  default_measure = msr_torch("mse", function(truth, response) mean((truth - response)^2)))
msr("torch.default")
#> 
#> ── <MeasureTorchDefault> (torch.default): Default Measure for a TaskTorch ──────
#> • Packages: mlr3
#> • Range: [-Inf, Inf]
#> • Minimize: NA
#> • Average: macro
#> • Parameters: list()
#> • Properties: requires_task and obs_loss
#> • Predict type: response
#> • Predict sets: test
#> • Aggregator: mean()
```
