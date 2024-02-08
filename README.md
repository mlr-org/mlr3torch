
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

## Status

`mlr3torch` is currently still in its experimental phase and many things
are missing. Not everything will work yet and the API might change
without notice.

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

This learner can for be resampled, benchmarked or tuned as any other
learner.

``` r
resample(
  task       = tsk("iris"),
  learner    = learner_mlp,
  resampling = rsmp("holdout")
)
#> <ResampleResult> with 1 resampling iterations
#>  task_id  learner_id resampling_id iteration warnings errors
#>     iris classif.mlp       holdout         1        0      0
```

Below, we construct the same architecture using `PipeOpTorch` objects.
The first pipeop – a `PipeOpTorchIngress` – defines the entrypoint of
the network. All subsequent pipeops define the neural network layers.

``` r
architecture = po("torch_ingress_num") %>>%
  po("nn_linear", out_features = 20) %>>%
  po("nn_relu") %>>%
  po("nn_head")
```

To turn this into a learner, we configure the loss, optimizer, callbacks
and the training arguments.

``` r
graph_mlp = architecture %>>%
  po("torch_loss", loss = t_loss("cross_entropy")) %>>%
  po("torch_optimizer", optimizer = t_opt("adam", lr = 0.1)) %>>%
  po("torch_callbacks", callbacks = t_clbk("history")) %>>%
  po("torch_model_classif", batch_size = 16, epochs = 50, device = "cpu")

graph_mlp
#> Graph with 8 PipeOps:
#>                   ID         State            sccssors         prdcssors
#>               <char>        <char>              <char>            <char>
#>    torch_ingress_num <<UNTRAINED>>           nn_linear                  
#>            nn_linear <<UNTRAINED>>             nn_relu torch_ingress_num
#>              nn_relu <<UNTRAINED>>             nn_head         nn_linear
#>              nn_head <<UNTRAINED>>          torch_loss           nn_relu
#>           torch_loss <<UNTRAINED>>     torch_optimizer           nn_head
#>      torch_optimizer <<UNTRAINED>>     torch_callbacks        torch_loss
#>      torch_callbacks <<UNTRAINED>> torch_model_classif   torch_optimizer
#>  torch_model_classif <<UNTRAINED>>                       torch_callbacks

graph_lrn = as_learner(graph_mlp)
graph_lrn$id = "graph_mlp"

resample(
  task       = tsk("iris"),
  learner    = graph_lrn,
  resampling = rsmp("holdout")
)
#> <ResampleResult> with 1 resampling iterations
#>  task_id learner_id resampling_id iteration warnings errors
#>     iris  graph_mlp       holdout         1        0      0
```

To work with generic tensors, the `lazy_tensor` type can be used. It
wraps a `torch::dataset`, but allows to preproress the data using
`PipeOp` objects, just like tabular data.

``` r
# load the predefined mnist task
task = tsk("mnist")
task$head()
#>     label           image
#>    <fctr>   <lazy_tensor>
#> 1:      5 <tnsr[1x28x28]>
#> 2:      0 <tnsr[1x28x28]>
#> 3:      4 <tnsr[1x28x28]>
#> 4:      1 <tnsr[1x28x28]>
#> 5:      9 <tnsr[1x28x28]>
#> 6:      2 <tnsr[1x28x28]>

# Resize the images to 5x5
po_resize = po("trafo_resize", size = c(5, 5))
task_reshaped = po_resize$train(list(task))[[1L]]

task_reshaped$head()
#>     label         image
#>    <fctr> <lazy_tensor>
#> 1:      5 <tnsr[1x5x5]>
#> 2:      0 <tnsr[1x5x5]>
#> 3:      4 <tnsr[1x5x5]>
#> 4:      1 <tnsr[1x5x5]>
#> 5:      9 <tnsr[1x5x5]>
#> 6:      2 <tnsr[1x5x5]>

# The tensors are loaded and preprocessed only when materialized

materialize(
  task_reshaped$data(1, cols = "image")[[1L]],
  rbind = TRUE
)
#> torch_tensor
#> (1,1,.,.) = 
#>     0.0000    0.0000    0.0000    0.0000    0.0000
#>     0.0000  200.9199  228.2500    8.2000    0.0000
#>     0.0000    0.0000  196.7500    0.0000    0.0000
#>     0.0000    0.0000  194.9500  147.4199    0.0000
#>     0.0000   64.8303    0.0000    0.0000    0.0000
#> [ CPUFloatType{1,1,5,5} ]
```

## Feature Overview

- Off-the-shelf architectures are readily available as `mlr3::Learner`s.
- Custom learners can be defined using the `Graph` language from
  `mlr3pipelines`
- The package supports tabular data, as well as generic tensors via the
  `lazy_tensor` type
- Multi-modal data can be handled conveniently, as `lazy_tensor` objects
  can be stored alongside tabular data.
- It is possible to customize the training process via (predefined or
  custom) callbacks.
- The package is fully integrated into the `mlr3` ecosystem.

## Documentation

Coming soon.

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
