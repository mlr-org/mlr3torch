# Create a Measure for a Generic Torch Task

Short form for constructing a
[`MeasureTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_measures_torch.md).
See the *Custom Learning Problems* article for how to create and use
such measures.

## Usage

``` r
msr_torch(
  id,
  fun,
  minimize = NA,
  range = c(-Inf, Inf),
  predict_type = "response",
  properties = character(),
  label = NA_character_,
  obs_loss = NULL
)
```

## Arguments

- id:

  (`character(1)`)  
  The id of the measure.

- fun:

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

- minimize:

  (`logical(1)`)  
  Whether a smaller score is better. `NA` (default) means the direction
  is unknown.

- range:

  (`numeric(2)`)  
  The range of possible scores.

- predict_type:

  (`character(1)`)  
  The predict type the measure requires: `"response"` (default),
  `"prob"`, `"se"` or `"lazy_tensor"`. A measure asking for one it did
  not declare here still receives it, if the prediction has it.

- properties:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Properties of the measure, see
  [`Measure`](https://mlr3.mlr-org.com/reference/Measure.html). The
  `"requires_task"`, `"requires_learner"`, `"requires_train_set"` and
  `"weights"` properties are added automatically when `fun` declares the
  corresponding argument. `"requires_model"` is not: a `learner`
  argument only says that the learner object is needed, and `mlr3` hands
  that over even when the model was not stored – after a
  [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) with
  `store_models = FALSE`, `learner$network` is then `NULL` and a measure
  reading it scores whatever an empty model gives. Pass
  `properties = "requires_model"` yourself whenever the measure reaches
  for the trained network, so that `mlr3` refuses to score instead.

- label:

  (`character(1)`)  
  The label of the measure.

- obs_loss:

  (`function()` or `NULL`)  
  The per-observation loss. Declared like `fun`, except that `train_set`
  is not available here, and if specified adds the `"obs_loss"`
  property. It must return one number per observation: a multi-target
  loss reduces over the targets
  ([`rowMeans()`](https://rdrr.io/r/base/colSums.html)), not over the
  observations ([`mean()`](https://rdrr.io/r/base/mean.html)), and
  returning a single number is an error rather than a column of that
  number repeated.

## Value

[`MeasureTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_measures_torch.md)

## Examples

``` r
m = msr_torch("hamming", function(truth, response) mean(as.matrix(truth) != response))
m$properties
#> character(0)

# with a per-observation loss
m = msr_torch("mse", function(truth, response) mean((truth - response)^2),
  obs_loss = function(truth, response) (truth - response)^2)
m$properties
#> [1] "obs_loss"
```
