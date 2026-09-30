# Custom Learning Problems

This article will show how to use {mlr3torch} for tasks that go beyond
the classification and regression tasks that are supported out of the
box. This is possible via the generic `TaskTorch`, which allows you to
use mlr3torch with arbitrary modeling problems. The price of this
flexibility is fewer compatibility checks and thus more responsibility
on the user, see [Drawbacks of `TaskTorch`](#sec-drawbacks) for more
information. In this article we will first show how to use `TaskTorch`
in a multi-label classification problem and then how it can be used for
an unsupervised autoencoder.

## Multi-label Classification

As an example, we will train a learner on a multi-label classification
problem which consists of \\k\\ binary prediction problems where
multiple labels can be true. Below, we generate some synthetic data for
it and want to predict from the five `x` features whether it’s sunny,
warm, and windy.

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

set.seed(314)
n = 500
dat = data.table(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n), x5 = rnorm(n))
dat[, `:=`(
  sunny = x1 + x2 > 0,
  warm = x2 * x3 > 0,
  windy = x4 - x5 > 0
)]
```

In order to create a `TaskTorch` for this modeling problem, we want to
first specify:

1.  The default method for converting torch predictions to an
    [`mlr3::Prediction`](https://mlr3.mlr-org.com/reference/Prediction.html)
    object,
2.  The default
    [`mlr3::Measure`](https://mlr3.mlr-org.com/reference/Measure.html)
    for scoring these predictions, and
3.  How many output units a network for this task needs.

For our problem, we want to accept both response (class) predictions and
probability predictions, and we assume that the network outputs
per-class logits. Any learner that we create for this task can also
deviate from this default encoding.

``` r

weather_encoder = function(task, network_output, predict_type) {
  prob = as.matrix(nnf_sigmoid(network_output)$cpu())
  colnames(prob) = task$target_names
  list(response = prob > 0.5, prob = if (predict_type == "prob") prob)
}
```

For more information on what a prediction may hold and how it is tabled,
see
[`?PredictionTorch`](https://mlr3torch.mlr-org.com/dev/reference/PredictionTorch.md).
As the default measure, we use the *Hamming loss* and construct it via
`msr_torch`, see its help page for more information.

``` r

msr_hamming = msr_torch("multilabel.hamming",
  function(truth, response) mean(as.matrix(truth) != response),
  range = c(0, 1), minimize = TRUE
)
msr_hamming
#> 
#> ── <MeasureTorch> (multilabel.hamming) ─────────────────────────────────────────
#> • Packages: mlr3
#> • Range: [0, 1]
#> • Minimize: TRUE
#> • Average: macro
#> • Parameters: list()
#> • Properties: -
#> • Predict type: response
#> • Predict sets: test
#> • Aggregator: mean()
```

Below, we create the `TaskTorch`, where the third of these is the
`output_dim` argument, a function of the task.

``` r

tsk_weather = as_task_torch(dat, target = c("sunny", "warm", "windy"), id = "weather",
  output_dim = function(task) length(task$target_names),
  default_encoder = weather_encoder,
  default_measure = msr_hamming
)
tsk_weather
#> 
#> ── <TaskTorch> (500x8) ─────────────────────────────────────────────────────────
#> • Target: sunny, warm, and windy
#> • Properties: -
#> • Features (5):
#>   • dbl (5): x1, x2, x3, x4, x5

tsk_weather$truth(1:5)
#>     sunny   warm  windy
#>    <lgcl> <lgcl> <lgcl>
#> 1:   TRUE  FALSE   TRUE
#> 2:   TRUE  FALSE   TRUE
#> 3:  FALSE  FALSE  FALSE
#> 4:   TRUE   TRUE   TRUE
#> 5:   TRUE  FALSE   TRUE
tsk_weather$target_names
#> [1] "sunny" "warm"  "windy"
tsk_weather$feature_names
#> [1] "x1" "x2" "x3" "x4" "x5"
```

The `output_dim` field is used via the
[`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md)
generic:

``` r

output_dim_for(tsk_weather)
```

Below, we define the architecture where the output dimension is obtained
using
[`output_dim_for()`](https://mlr3torch.mlr-org.com/dev/reference/output_dim_for.md).

``` r

nn_mlp = nn_module("nn_mlp",
  initialize = function(task, latent) {
    self$net = nn_sequential(
      nn_linear(length(task$feature_names), latent), nn_relu(),
      nn_linear(latent, latent), nn_relu(),
      nn_linear(latent, output_dim_for(task))
    )
  },
  # the argument name matches the name of the ingress token below
  forward = function(input) self$net(input)
)
```

To turn this into a learner, we additionally need a loss and a way to
construct the batches. For the features, we use a standard numeric
encoder via
[`ingress_num()`](https://mlr3torch.mlr-org.com/dev/reference/ingress_num.md),
so we only specify the `target_batchgetter` to also encode the targets
numerically. The data argument of the batchgetter is
`task$data(batch_ids, task$target_names)` and it must return a torch
tensor. For the loss, we use the binary cross entropy, built from the
corresponding torch loss:

``` r

loss_bce = as_torch_loss(nn_bce_with_logits_loss, id = "bce")
loss_bce
#> <TorchLoss:bce> bce
#> * Generator: nn_bce_with_logits_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: classif,regr,torch

# the loss wants a float tensor of zeros and ones, so that is what the learner builds
weather_batchgetter = function(data) torch_tensor(as.matrix(data), dtype = torch_float())

lrn_weather = lrn("torch.module",
  module_generator = nn_mlp,
  ingress_tokens = list(input = ingress_num()),
  loss = loss_bce,
  target_batchgetter = weather_batchgetter,
  predict_types = c("response", "prob"),
  latent = 32,
  epochs = 5, batch_size = 32, opt.lr = 0.01
)
lrn_weather$predict_type = "prob"
```

Next, we train the learner on two thirds of the data and make
predictions on the remaining observations:

``` r

split = partition(tsk_weather)
lrn_weather$train(tsk_weather, split$train)
pred = lrn_weather$predict(tsk_weather, split$test)
pred
#> 
#> ── <PredictionTorch> for 165 observations: ─────────────────────────────────────
#>  row_ids truth.sunny truth.warm truth.windy   response   prob.sunny
#>        1        TRUE      FALSE        TRUE <array[3]> 9.884939e-01
#>        2        TRUE      FALSE        TRUE <array[3]> 9.999373e-01
#>        3       FALSE      FALSE       FALSE <array[3]> 3.701719e-04
#>      ---         ---        ---         ---        ---          ---
#>      494        TRUE       TRUE        TRUE <array[3]> 9.996092e-01
#>      496        TRUE       TRUE       FALSE <array[3]> 9.694263e-01
#>      500       FALSE       TRUE        TRUE <array[3]> 1.118472e-07
#>     prob.warm  prob.windy
#>  0.0009286193 0.995082736
#>  0.0365176164 0.999844551
#>  0.2753116488 0.015916388
#>           ---         ---
#>  0.9642977118 0.998322666
#>  0.8094004989 0.004069732
#>  0.9803524017 0.992626727
```

The prediction is scored like any other. Note that a
[`mlr3::Measure`](https://mlr3.mlr-org.com/reference/Measure.html)
differs from a `TorchLoss`: it is computed from the encoded prediction,
whereas the torch loss operates on the raw network output and the
tensor-encoded target.

``` r

pred$score(msr_hamming)
#> multilabel.hamming 
#>         0.07878788
```

It is also possible to get the raw network outputs as a `lazy_tensor`:

``` r

lrn_logits = lrn_weather$clone(deep = TRUE)
lrn_logits$predict_type = "lazy_tensor"

pred_logits = lrn_logits$predict(tsk_weather, row_ids = split$test[1:3])
pred_logits
#> 
#> ── <PredictionTorch> for 3 observations: ───────────────────────────────────────
#>  row_ids truth.sunny truth.warm truth.windy lazy_tensor
#>        1        TRUE      FALSE        TRUE   <tnsr[3]>
#>        2        TRUE      FALSE        TRUE   <tnsr[3]>
#>        3       FALSE      FALSE       FALSE   <tnsr[3]>
materialize(pred_logits$lazy_tensor, rbind = TRUE)
#> torch_tensor
#>  4.4533 -6.9809  5.3101
#>  9.6765 -3.2728  8.7694
#> -7.9012 -0.9678 -4.1244
#> [ CPUFloatType{3,3} ]
```

Of course, we can also resample the learner.

``` r

rr = resample(tsk_weather, lrn_weather, rsmp("cv", folds = 3))
rr$aggregate()
#> torch.default 
#>     0.1033876
```

We can also construct the learner as a `Graph`. Here, we have to specify
the `target_batchgetter` in the `PipeOpTorchModel`. Note that
`nn("head")` also relies on `output_dim_for` to define the last layer.

``` r

architecture = po("torch_ingress_num") %>>%
  nn("linear_1", out_features = 20) %>>%
  nn("relu_1") %>>%
  nn("linear_2", out_features = 20) %>>%
  nn("relu_2") %>>%
  nn("head") %>>%
  po("torch_loss", loss_bce) %>>%
  po("torch_optimizer", "adam", lr = 0.01) %>>%
  po("torch_model", batch_size = 32, epochs = 50,
    target_batchgetter = weather_batchgetter)

glrn_weather = as_learner(architecture)
glrn_weather$train(tsk_weather, row_ids = split$train)
glrn_weather$predict(tsk_weather, row_ids = split$test)$score(msr_hamming)
#> multilabel.hamming 
#>         0.04646465
```

## A Simple Autoencoder

A `TaskTorch` is supervised or unsupervised depending only on whether
you gave it target columns. Nothing beyond that is assumed about the
structure of the problem.

A task with no targets has no target element in its batches at all, so
the loss is called as `loss(y_hat)`, with no second argument.

If the target of a batch is a function of its *input*, as it is for an
autoencoder reconstructing its input, a denoising or masked objective,
or contrastive pretraining, then the learner’s `target_batchgetter` may
declare an `x` argument, which receives the feature tensors of the
batch. An autoencoder over the numeric features of `iris` is then:

``` r

iris_scaled = as.data.table(scale(iris[, 1:4]))

tsk_ae = as_task_torch(iris_scaled, id = "iris_ae",
  output_dim = function(task) length(task$feature_names),
  default_encoder = function(task, network_output, predict_type) {
    response = as.matrix(network_output$cpu())
    colnames(response) = task$feature_names
    list(response = response)
  }
)
tsk_ae
#> 
#> ── <TaskTorch> (150x4) ─────────────────────────────────────────────────────────
#> • Target:
#> • Properties: -
#> • Features (4):
#>   • dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width
```

Such a task has no `truth`, so its measure reads the ground truth from
the task.
[`msr_torch()`](https://mlr3torch.mlr-org.com/dev/reference/msr_torch.md)
arranges that for any function that declares a `task` argument:

``` r

msr_recon = msr_torch("reconstruction", function(task, prediction) {
  truth = as.matrix(task$data(rows = prediction$row_ids, cols = task$feature_names))
  mean((truth - prediction$response)^2)
}, range = c(0, Inf))
```

The network is an ordinary autoencoder, and the loss compares its output
to the target that the batchgetter produced:

``` r

nn_ae = nn_module("nn_ae",
  initialize = function(task) {
    d_in = length(task$feature_names)
    self$encoder = nn_sequential(
      nn_linear(d_in, 16),
      nn_relu(),
      nn_linear(16, 2)
    )
    self$decoder = nn_sequential(
      nn_linear(2, 16),
      nn_relu(),
      nn_linear(16, d_in)
    )
  },
  forward = function(input) self$decoder(self$encoder(input))
)

lrn_ae = lrn("torch.module",
  module_generator = nn_ae,
  ingress_tokens = list(input = ingress_num()),
  loss = t_loss("mse"),
  # a reconstruction objective: the target of a batch is the batch's own input
  target_batchgetter = function(data, x) x[[1L]],
  epochs = 100, batch_size = 32, opt.lr = 0.01
)

lrn_ae$train(tsk_ae)
lrn_ae$predict(tsk_ae)$score(msr_recon, task = tsk_ae)
#> reconstruction 
#>      0.0316384
```

### Predictions as Tensors

In some cases, you mighth want to directly access the raw tensor
predictions instead of an encoded variant. This is possible via the
`"lazy_tensor"` predict type:

``` r

lrn_raw = lrn_ae$clone(deep = TRUE)
lrn_raw$predict_type = "lazy_tensor"

pred_raw = lrn_raw$predict(tsk_ae)
pred_raw
#> 
#> ── <PredictionTorch> for 150 observations: ─────────────────────────────────────
#>  row_ids lazy_tensor
#>        1   <tnsr[4]>
#>        2   <tnsr[4]>
#>        3   <tnsr[4]>
#>      ---         ---
#>      148   <tnsr[4]>
#>      149   <tnsr[4]>
#>      150   <tnsr[4]>
```

You can convert this to an actual `torch_tensor` by materializing it:

``` r

reconstruction = materialize(pred_raw$lazy_tensor, rbind = TRUE)
reconstruction$shape
#> [1] 150   4
```

Note that this prediction can currently not be correctly saved via
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) as it holds external
pointers.

You can also directly access the network, e.g. to get the latent
embeddings:

``` r

ds = lrn_ae$dataset(tsk_ae)
batch = ds$.getbatch(1:5)

network = lrn_ae$network
network$eval()
with_no_grad(network$encoder(batch$x$input))$cpu()
#> torch_tensor
#> -2.4929  0.0720
#> -2.3052  0.9233
#> -2.5667  0.7435
#> -2.5191  0.9110
#> -2.6124 -0.0263
#> [ CPUFloatType{5,2} ]
```

## Drawbacks of `TaskTorch`

Now that we have shown the capabilities of `TaskTorch`, it is also
important to highlight the drawbacks of the approach. In {mlr3}, objects
are annotated with meta-information that describes their properties and
capabilities. For example, training a classification learner on a
regression task raises a clear error message:

``` r

lrn("classif.featureless")$train(tsk("mtcars"))
#> Error:
#> ! 
#> ✖ Type 'regr' of <TaskRegr:mtcars> does not match type 'classif' of
#>   <LearnerClassifFeatureless:classif.featureless>
#> → Class: Mlr3ErrorInput
```

This is because they have different `task_type`s. Because the `"torch"`
task type can represent an arbitrary task type (that is what it is
designed for), we can’t distinguish between the multi-label
classification task and the reconstruction problem. E.g., training the
autoencoder on the weather data is simply undefined behavior. The same
also holds true for measures. As such, the `TaskTorch` is a *quick and
dirty* way to approach a modeling problem. The alternative is to
properly create a new task type, which requires implementing sub-classes
(such as `TaskRegr`, `MeasureRegr`, etc.) and registering the new task
type with {mlr3}.
