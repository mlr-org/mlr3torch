
<!-- README.md is generated from README.Rmd. Please edit that file -->

# mlr3torch <img src="man/figures/logo.svg" align="right" width = "120" />

Package website: [dev](https://mlr3torch.mlr-org.com/)

Deep Learning with torch and mlr3.

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![r-cmd-check](https://github.com/mlr-org/mlr3torch/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/mlr-org/mlr3torch/actions/workflows/r-cmd-check.yml)
[![CRAN
status](https://www.r-pkg.org/badges/version/mlr3torch)](https://CRAN.R-project.org/package=mlr3torch)
[![StackOverflow](https://img.shields.io/badge/stackoverflow-mlr3-orange.svg)](https://stackoverflow.com/questions/tagged/mlr3)
[![Mattermost](https://img.shields.io/badge/chat-mattermost-orange.svg)](https://lmmisld-lmu-stats-slds.srv.mwn.de/mlr_invite/)
<!-- badges: end -->

## Installation

``` r
# Install the development version from GitHub:
pak::pak("mlr-org/mlr3torch")
# You also need to install torch:
torch::install_torch()
```

More information about installing `torch` can be found
[here](https://torch.mlverse.org/docs/articles/installation.html).

## What is mlr3torch?

`mlr3torch` is a deep learning framework for the
[`mlr3`](https://mlr-org.com) ecosystem built on top of
[`torch`](https://torch.mlverse.org/). It allows to easily build, train
and evaluate deep learning models in a few lines of codes, without
needing to worry about low-level details. Off-the-shelf learners are
readily available, but custom architectures can be defined by connecting
`PipeOpTorch` operators in an `mlr3pipelines::Graph`.

Using predefined learners such as a simple multi layer perceptron (MLP)
works just like any other mlr3 `Learner`.

``` r
library(mlr3torch)
learner_mlp = lrn("classif.mlp",
  # defining network parameters
  activation     = nn_relu,
  neurons        = c(20, 20),
  # training parameters
  batch_size     = 16,
  epochs         = 50,
  device         = "cpu",
  # Proportion of data to use for validation
  validate = 0.3,
  # Defining the optimizer, loss, and callbacks
  optimizer      = t_opt("adam", lr = 0.1),
  loss           = t_loss("cross_entropy"),
  callbacks      = t_clbk("history"), # this saves the history in the learner
  # Measures to track
  measures_valid = msrs(c("classif.logloss", "classif.ce")),
  measures_train = msrs(c("classif.acc")),
  # predict type (required by logloss)
  predict_type = "prob"
)
```

Below, we train this learner on the sonar example task:

``` r
learner_mlp$train(tsk("sonar"))
```

Next, we construct the same architecture using `PipeOpTorch` objects.
The first pipeop – a `PipeOpTorchIngress` – defines the entrypoint of
the network. All subsequent pipeops define the neural network layers.

``` r
architecture = po("torch_ingress_num") %>>%
  po("nn_linear", out_features = 20) %>>%
  po("nn_relu") %>>%
  po("nn_head")
```

To turn this into a learner, we configure the loss, optimizer, callbacks
as well as the training arguments.

``` r
graph_mlp = architecture %>>%
  po("torch_loss", loss = t_loss("cross_entropy")) %>>%
  po("torch_optimizer", optimizer = t_opt("adam", lr = 0.1)) %>>%
  po("torch_callbacks", callbacks = t_clbk("history")) %>>%
  po("torch_model_classif",
    batch_size = 16, epochs = 50, device = "cpu")

graph_lrn = as_learner(graph_mlp)
```

To work with generic tensors, the `lazy_tensor` type can be used. It
wraps a `torch::dataset`, but allows to preprocess the data (lazily)
using `PipeOp` objects. Below, we flatten the MNIST task, so we can then
train a multi-layer perceptron on it. Note that this does *not*
transform the data in-memory, but is only applied when the data is
actually loaded.

``` r
# load the predefined mnist task
mnist = tsk("mnist")
mnist$head(3L)
#>     label           image
#>    <fctr>   <lazy_tensor>
#> 1:      5 <tnsr[1x28x28]>
#> 2:      0 <tnsr[1x28x28]>
#> 3:      4 <tnsr[1x28x28]>

# Flatten the images
flattener = po("trafo_reshape", shape = c(-1, 28 * 28))
mnist_flat = flattener$train(list(mnist))[[1L]]

mnist_flat$head(3L)
#>     label         image
#>    <fctr> <lazy_tensor>
#> 1:      5   <tnsr[784]>
#> 2:      0   <tnsr[784]>
#> 3:      4   <tnsr[784]>
```

To actually access the tensors, we can call `materialize()`. We only
show a slice of the resulting tensor for readability:

``` r
materialize(
  mnist_flat$data(1:2, cols = "image")[[1L]],
  rbind = TRUE
)[1:2, 1:4]
#> torch_tensor
#>  0  0  0  0
#>  0  0  0  0
#> [ CPUFloatType{2,4} ]
```

Below, we define a more complex architecture that has one single input
which is a `lazy_tensor`. For that, we first define a single residual
block:

``` r
layer = list(
  po("nop"),
  po("nn_linear", out_features = 50L) %>>%
    po("nn_dropout") %>>% po("nn_relu")
) %>>% po("nn_merge_sum")
```

Next, we create a neural network that takes as input a `lazy_tensor`
(`po("torch_ingress_num")`). It first applies a linear layer and then
repeats the above layer using the special `PipeOpTorchBlock`, followed
by the network’s head. After that, we configure the loss, optimizer and
the training parameters. Note that `po("nn_linear_0")` is equivalent to
`po("nn_linear", id = "nn_linear_0")` and we need this here to avoid ID
clashes with the linear layer from `po("nn_block")`.

``` r
deep_network = po("torch_ingress_ltnsr") %>>%
  po("nn_linear_0", out_features = 50L) %>>%
  po("nn_block", layer, n_blocks = 5L) %>>%
  po("nn_head") %>>%
  po("torch_loss", loss = t_loss("cross_entropy")) %>>%
  po("torch_optimizer", optimizer = t_opt("adam")) %>>%
  po("torch_model_classif",
    epochs = 100L, batch_size = 32
  )
```

Next, we prepend the preprocessing step that flattens the images so we
can directly apply this learner to the unflattened MNIST task.

``` r
deep_learner = as_learner(
  flattener %>>% deep_network
)
deep_learner$id = "deep_network"
```

In order to keep track of the performance during training, we use 20% of
the data and evaluate it using classification accuracy.

``` r
set_validate(deep_learner, 0.2)
deep_learner$param_set$set_values(
  torch_model_classif.measures_valid = msr("classif.acc")
)
```

All that is left is to train the learner:

``` r
deep_learner$train(mnist)
```

## Feature Overview

- Off-the-shelf architectures are readily available as `mlr3::Learner`s.
- Currently, supervised regression and classification is supported.
- Custom learners can be defined using the `Graph` language from
  `mlr3pipelines`.
- The package supports tabular data, as well as generic tensors via the
  `lazy_tensor` type.
- Multi-modal data can be handled conveniently, as `lazy_tensor` objects
  can be stored alongside tabular data.
- It is possible to customize the training process via (predefined or
  custom) callbacks.
- The package is fully integrated into the `mlr3` ecosystem.
- Neural network architectures, as well as their hyperparameters can be
  easily tuned via `mlr3tuning` and friends.

## Documentation

- Start by reading one of the vignettes on the package website!

## Contributing:

- To run the tests one needs to run set the environment variable
  `TEST_TORCH = 1`, e.g. by adding it to `.Renviron`.

## Acknowledgements

- Without the great R package `torch` none of this would have been
  possible.
- The names for the callback stages are taken from
  [luz](https://mlverse.github.io/luz/), another high-level deep
  learning framework for R `torch`.
- Building neural networks using `PipeOpTorch` operators is inspired by
  [keras](https://keras.io/).

## Bugs, Questions, Feedback

*mlr3torch* is a free and open source software project that encourages
participation and feedback. If you have any issues, questions,
suggestions or feedback, please do not hesitate to open an “issue” about
it on the GitHub page!

In case of problems / bugs, it is often helpful if you provide a
“minimum working example” that showcases the behaviour (but don’t worry
about this if the bug is obvious).

Please understand that the resources of the project are limited:
response may sometimes be delayed by a few days, and some feature
suggestions may be rejected if they are deemed too tangential to the
vision behind the project.
