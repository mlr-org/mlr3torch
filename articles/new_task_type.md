# Adding a New Task Type

The [Custom Learning
Problems](https://mlr3torch.mlr-org.com/articles/custom_task_type.md)
article shows how to use the generic `TaskTorch` for learning problems
beyond classification and regression. This is the best approach for
one-off modeling problems that are not supported out of the box by
{mlr3torch}. If you instead want to create a task type that is fully
integrated into the {mlr3} ecosystem, e.g. for creating an extension
package, this article will show you how to do that. If you want to do
so, it is best to get in touch with us to discuss the implementation and
provide guidance.

As a running example, we use a very simple task type `"count"` for count
data, i.e., Poisson regression. Note that we are showing a minimal
implementation for illustrative purposes that omits some input checks to
keep the example concise.

``` r

library(mlr3torch)
#> Loading required package: mlr3
#> Loading required package: mlr3pipelines
#> Loading required package: torch
library(data.table)
#> 
#> Attaching package: 'data.table'
#> The following object is masked from 'package:base':
#> 
#>     %notin%
library(R6)
```

## Registering the Task Type with mlr3

A task type in {mlr3} consists of a `Task`, a `Prediction` and a
`Measure` class (and a `Learner`, which we cover afterwards), which have
to be registered in `mlr_reflections`. `mlr_reflections` is the central
registry of {mlr3}: an environment with lookup tables that define what
is valid for each task type, e.g., the class names, the column roles,
the properties, and the predict types. All of its entries are documented
in
[`mlr_reflections`](https://mlr3.mlr-org.com/reference/mlr_reflections.html).
Instead of hard-coding the task types, {mlr3} looks them up in these
tables, so registering a new task type there makes it available
throughout {mlr3}. These are independent of {mlr3torch} and required for
any new task type in {mlr3}, so we keep them as short as possible here.
If the task type is already defined by another package, e.g. survival
tasks in {mlr3survival}, you can skip this section and continue with
[Supporting the Task Type in
mlr3torch](#supporting-the-task-type-in-mlr3torch). For more
information, always consult the documentation of the classes we inherit
from, such as
[`?mlr3::TaskSupervised`](https://mlr3.mlr-org.com/reference/TaskSupervised.html),
[`?mlr3::Prediction`](https://mlr3.mlr-org.com/reference/Prediction.html),
[`?mlr3::Measure`](https://mlr3.mlr-org.com/reference/Measure.html), and
[`?mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md).
For our count task, the target is a single integer column and we are
inheriting from `TaskSupervised` because it is a supervised learning
problem.

``` r

TaskCount = R6Class("TaskCount", inherit = TaskSupervised,
  public = list(
    initialize = function(id, backend, target, label = NA_character_) {
      super$initialize(id = id, task_type = "count", backend = backend, target = target,
        label = label)
    },
    truth = function(rows = NULL) super$truth(rows)[[1L]]
  )
)
```

The only property we need to check for the task’s metadata is that the
target is a single integer column. We do this by implementing
`task_check_col_roles.TaskCount()`:

``` r

task_check_col_roles.TaskCount = function(task, new_roles, ...) {
  target = new_roles$target
  if (length(target) > 1L) {
    stop("Count tasks must have exactly one target column")
  }
  if (length(target) && task$col_info[list(target), "type", on = "id"][[1L]] != "integer") {
    stop(sprintf("Target column '%s' must be an integer column", target))
  }
  NextMethod()
}
```

Next, we define the structure of the prediction objects for this task
type. A prediction for this task type contains the predicted rate as the
`response`, i.e., the expected count, which is the parameter \\\lambda\\
of the Poisson distribution. For simplicity, we only support this single
predict type `"response"`, but a task type can have multiple predict
types, e.g., `"response"` and `"se"` for regression.

Internally, {mlr3} mostly works with a lightweight representation of
predictions: an S3 object of class `PredictionData<Type>`, i.e., a named
list with class `c("PredictionDataCount", "PredictionData")`. It is also
what is stored in the user-facing `Prediction` object, which is an `R6`
class and is returned, e.g., by `learner$predict()`. Note that the
`$response()` method needs to handle a non-existing target column to
support prediction in the inference setting where no ground truth is
available.

``` r

new_prediction_data_count = function(row_ids, truth, response) {
  structure(list(row_ids = row_ids, truth = truth, response = response),
    class = c("PredictionDataCount", "PredictionData"))
}

PredictionCount = R6Class("PredictionCount", inherit = Prediction,
  public = list(
    initialize = function(task = NULL, row_ids = task$row_ids, truth = task$truth(),
      response = NULL, check = TRUE) {
      pdata = new_prediction_data_count(row_ids, truth, response)
      if (check) pdata = check_prediction_data(pdata)
      self$task_type = "count"
      self$data = pdata
      self$predict_types = "response"
    }
  ),
  active = list(
    response = function() {
      if (is.null(self$data$response)) rep(NA_real_, length(self$data$row_ids)) else self$data$response
    }
  )
)
```

For the prediction data, a few S3 generics of {mlr3} have to be
implemented: checking, filtering, and combining the prediction data,
finding missing predictions, and converting it to a `Prediction`. In
addition,
[`create_empty_prediction_data()`](https://mlr3.mlr-org.com/reference/PredictionData.html)
creates the prediction data when a learner predicts on a task without
rows; it dispatches on the class of the task. Their implementations are
straightforward, so we hide them below.

Implementation of the S3 methods

``` r

as.data.table.PredictionCount = function(x, ...) {
  data.table(row_ids = x$row_ids, truth = x$truth, response = x$response)
}

check_prediction_data.PredictionDataCount = function(pdata, ...) {
  pdata$row_ids = checkmate::assert_integerish(pdata$row_ids, coerce = TRUE)
  pdata$response = checkmate::assert_numeric(unname(pdata$response), lower = 0,
    len = length(pdata$row_ids), null.ok = TRUE)
  pdata
}

is_missing_prediction_data.PredictionDataCount = function(pdata, ...) {
  pdata$row_ids[is.na(pdata$response)]
}

filter_prediction_data.PredictionDataCount = function(pdata, row_ids, ...) {
  keep = pdata$row_ids %in% row_ids
  pdata[c("row_ids", "truth", "response")] = lapply(pdata[c("row_ids", "truth", "response")],
    function(x) x[keep])
  pdata
}

c.PredictionDataCount = function(..., keep_duplicates = TRUE) {
  dots = list(...)
  pdata = lapply(c(row_ids = "row_ids", truth = "truth", response = "response"),
    function(nm) unlist(lapply(dots, `[[`, nm)))
  if (!keep_duplicates) {
    keep = !duplicated(pdata$row_ids, fromLast = TRUE)
    pdata = lapply(pdata, function(x) x[keep])
  }
  new_prediction_data_count(pdata$row_ids, pdata$truth, pdata$response)
}

as_prediction.PredictionDataCount = function(x, check = FALSE, ...) {
  PredictionCount$new(row_ids = x$row_ids, truth = x$truth, response = x$response, check = check)
}

create_empty_prediction_data.TaskCount = function(task, learner) {
  new_prediction_data_count(integer(), integer(), numeric())
}
```

Other S3 methods for tasks, such as
[`head()`](https://rdrr.io/r/utils/head.html),
[`summary()`](https://rdrr.io/r/base/summary.html), or
[`partition()`](https://mlr3.mlr-org.com/reference/partition.html), are
inherited from `Task` and need no dedicated implementation in this case.
A package would usually also provide a converter `as_task_count()`,
analogous to
[`as_task_regr()`](https://mlr3.mlr-org.com/reference/as_task_regr.html),
which we omit here. To see which methods a complete implementation of a
task type should provide, look at an {mlr3} extension package that adds
a task type, such as {mlr3survival}.

Because this code is run as an article here, we need to manually
register the S3 methods. In a package, they are instead declared in the
`NAMESPACE` file via `S3method()` (e.g., using roxygen’s `@export` tag).

Registration of the S3 methods

``` r

for (generic in c("check_prediction_data", "is_missing_prediction_data",
  "filter_prediction_data", "c", "as_prediction")) {
  registerS3method(generic, "PredictionDataCount", get(paste0(generic, ".PredictionDataCount")))
}
registerS3method("as.data.table", "PredictionCount", as.data.table.PredictionCount)
for (generic in c("task_check_col_roles", "create_empty_prediction_data")) {
  registerS3method(generic, "TaskCount", get(paste0(generic, ".TaskCount")))
}
```

As the default measure, we use the (mean) Poisson deviance:

``` r

MeasureCountDeviance = R6Class("MeasureCountDeviance", inherit = Measure,
  public = list(
    initialize = function() {
      super$initialize(id = "count.deviance", task_type = "count",
        predict_type = "response", range = c(0, Inf), minimize = TRUE)
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      y = prediction$truth
      mu = prediction$response
      2 * mean(ifelse(y == 0, 0, y * log(y / mu)) - (y - mu))
    }
  )
)
mlr_measures$add("count.deviance", MeasureCountDeviance)
```

The last line adds the measure to the dictionary `mlr_measures` under
the key `"count.deviance"`, so that it can be constructed via
`msr("count.deviance")`. The default measure and the default loss of a
task type, which we register below, are also specified by their keys and
retrieved from these dictionaries.

Finally, we register the task type in `mlr_reflections`. Because our
learner is quite similar to a regression learner and we want to keep
things simple here, we reuse many of the settings from the regression
task type. The `package` column names the package that provides the task
type; `"mlr3count"` is a placeholder for the name of your extension
package.

``` r

mlr_reflections$task_types = setkeyv(rbind(mlr_reflections$task_types, data.table(
  type = "count", package = "mlr3count", task = "TaskCount", learner = "Learner",
  prediction = "PredictionCount", prediction_data = "PredictionDataCount",
  measure = "MeasureCountDeviance"
), fill = TRUE), "type")
for (registry in c("task_col_roles", "task_properties", "learner_properties",
  "measure_properties")) {
  mlr_reflections[[registry]]$count = mlr_reflections[[registry]]$regr
}
mlr_reflections$learner_predict_types$count = list(response = "response")
mlr_reflections$default_measures$count = "count.deviance"
```

## Supporting the Task Type in mlr3torch

{mlr3torch} has no separate learner classes per task type: all learners
inherit from `LearnerTorch`, which takes the `task_type` as a
construction argument. The task-type-specific behavior is defined by
three S3 generics, which dispatch on the class of the task:

- [`output_dim_for()`](https://mlr3torch.mlr-org.com/reference/output_dim_for.md):
  How many output units the network needs.
- [`get_target_batchgetter()`](https://mlr3torch.mlr-org.com/reference/get_target_batchgetter.md):
  How the target is converted to a tensor when loading the data.
- [`encode_prediction()`](https://mlr3torch.mlr-org.com/reference/encode_prediction.md):
  How the output of the network (the torch tensor returned by its
  `$forward()` method) is converted into the data of a prediction.

For Poisson regression, the network has a single output unit, which we
interpret as the log-rate.

``` r

output_dim_for.TaskCount = function(x, ...) 1L

get_target_batchgetter.TaskCount = function(task, ...) {
  function(data) torch_tensor(data[[1L]], dtype = torch_float32())$unsqueeze(2)
}

encode_prediction.TaskCount = function(task, network_output, predict_type, ...) {
  list(response = as.numeric(torch_exp(network_output)$cpu()))
}
```

Registration of the S3 methods

``` r

registerS3method("output_dim_for", "TaskCount", output_dim_for.TaskCount)
registerS3method("get_target_batchgetter", "TaskCount", get_target_batchgetter.TaskCount)
registerS3method("encode_prediction", "TaskCount", encode_prediction.TaskCount)
```

Next, we need a loss for the new task type. Unlike the measure, which
evaluates predictions in {mlr3} (e.g., in resampling or tuning), the
loss is a differentiable function of the network output and the target
tensor that is minimized during training. We add the Poisson negative
log-likelihood to the `mlr3torch_losses` dictionary and declare that it
can be used for the `"count"` task type.

``` r

mlr3torch_losses$add("poisson", function() {
  TorchLoss$new(torch_loss = nn_poisson_nll_loss, task_types = "count", id = "poisson",
    label = "Poisson NLL", man = "torch::nn_poisson_nll_loss")
})
```

Without further information, a learner for the new task type requires
the loss to be passed explicitly:

``` r

LearnerTorchMLP$new(task_type = "count")
#> Error:
#> ! There is no default loss for task type 'count', pass the `loss` explicitly.
```

To make it the default loss for the task type, we register its id in
`mlr_reflections$torch$default_losses`, which also holds the defaults
for classification and regression.

``` r

mlr_reflections$torch$default_losses$count = "poisson"
mlr_reflections$torch$default_losses
#> $classif
#> [1] "cross_entropy"
#> 
#> $regr
#> [1] "mse"
#> 
#> $count
#> [1] "poisson"
```

Finally, we add the learners we want to offer for the new task type to
the `mlr_learners` dictionary, here the multi-layer perceptron:

``` r

mlr_learners$add("count.mlp", function(...) LearnerTorchMLP$new(task_type = "count", ...))
```

This works for all learners whose architecture is agnostic to the task
type, i.e., which only size their output layer via
[`output_dim_for()`](https://mlr3torch.mlr-org.com/reference/output_dim_for.md).
Some learners have task-specific parts, however. For example, TabM
averages the predictions of its ensemble members differently for
classification and regression in its private `$.encode_prediction()`
method, so supporting a new task type requires overwriting this method.

## Using the New Task Type

We simulate some count data, where the rate depends on two features. An
actual extension package should also provide some example tasks, which
are added to the `mlr_tasks` dictionary so that they can be constructed
via [`tsk()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html).

``` r

set.seed(42)
n = 1000
dat = data.table(x1 = runif(n, -1, 1), x2 = runif(n, -1, 1))
dat$y = rpois(n, exp(0.5 + dat$x1 - 0.5 * dat$x2))
task = TaskCount$new("sim_count", backend = dat, target = "y")
task
#> 
#> ── <TaskCount> (1000x3) ────────────────────────────────────────────────────────
#> • Target: y
#> • Properties: -
#> • Features (2):
#>   • dbl (2): x1, x2
```

A non-integer target is rejected by
[`task_check_col_roles()`](https://mlr3.mlr-org.com/reference/task_check_col_roles.html):

``` r

TaskCount$new("invalid", backend = dat, target = "x1")
#> Error in `task_check_col_roles.TaskCount()`:
#> ! Target column 'x1' must be an integer column
```

The learner can now be constructed without specifying the loss, and the
measure defaults to the Poisson deviance.

``` r

learner = lrn("count.mlp", epochs = 50, batch_size = 32, neurons = 16, opt.lr = 0.01)
learner$loss
#> <TorchLoss:poisson> Poisson NLL
#> * Generator: nn_poisson_nll_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: count
```

Because the task type is registered with {mlr3}, everything just works,
e.g. the learner can be resampled like any other learner:

``` r

rr = resample(task, learner, rsmp("holdout"))
rr$aggregate()
#> count.deviance 
#>       1.132842
head(as.data.table(rr$prediction()))
#>    row_ids truth response
#>      <int> <int>    <num>
#> 1:       6     4 2.422913
#> 2:      12     1 2.762603
#> 3:      14     2 1.368224
#> 4:      15     0 2.350587
#> 5:      21     2 2.940447
#> 6:      24     2 5.090517
```

## Registration in a Package

In this article, we modified the registries `mlr_reflections`,
`mlr_measures`, `mlr3torch_losses`, and `mlr_learners` directly. If you
are working on a package, this should be done in the `.onLoad()` hook
instead, see {mlr3survival}’s `.onLoad()` implementation for an example.
This requires the package whose registries are extended to be in
`Imports`. If it is only in `Suggests`, e.g., to make {mlr3torch} an
optional dependency, it may not be loaded yet when your package is
loaded, so the registrations have to be done in a callback that runs
once it is loaded, which can be registered via
[`mlr3misc::register_namespace_callback()`](https://mlr3misc.mlr-org.com/reference/register_namespace_callback.html).
