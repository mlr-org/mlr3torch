
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
  # proportion of data for validation
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
# For GraphLearners, set_validate should be used to specify the validation data:
set_validate(graph_lrn, 0.3)

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
wraps a `torch::dataset`, but allows to preprocess the data using
`PipeOp` objects, just like tabular data. Below, we flatten the MNIST
task, so we can then train a multi-layer perceptron on it.

``` r
# load the predefined mnist task
mnist = tsk("mnist")
mnist$head()
#>     label           image
#>    <fctr>   <lazy_tensor>
#> 1:      5 <tnsr[1x28x28]>
#> 2:      0 <tnsr[1x28x28]>
#> 3:      4 <tnsr[1x28x28]>
#> 4:      1 <tnsr[1x28x28]>
#> 5:      9 <tnsr[1x28x28]>
#> 6:      2 <tnsr[1x28x28]>

# Flatten the images
flattener = po("trafo_reshape", shape = c(-1, 28 * 28))
mnist_flat = flattener$train(list(mnist))[[1L]]

mnist_flat$head()
#>     label         image
#>    <fctr> <lazy_tensor>
#> 1:      5   <tnsr[784]>
#> 2:      0   <tnsr[784]>
#> 3:      4   <tnsr[784]>
#> 4:      1   <tnsr[784]>
#> 5:      9   <tnsr[784]>
#> 6:      2   <tnsr[784]>

# The tensors are loaded and preprocessed only when materialized
materialize(
  mnist_flat$data(1, cols = "image")[[1L]],
  rbind = TRUE
)
#> torch_tensor
#> Columns 1 to 16   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 17 to 32   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 33 to 48   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 49 to 64   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 65 to 80   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 81 to 96   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 97 to 112   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 113 to 128   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 129 to 144   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 145 to 160   0    0    0    0    0    0    0    0    3   18   18   18  126  136  175   26
#> 
#> Columns 161 to 176 166  255  247  127    0    0    0    0    0    0    0    0    0    0    0    0
#> 
#> Columns 177 to 192  30   36   94  154  170  253  253  253  253  253  225  172  253  242  195   64
#> 
#> Columns 193 to 208   0    0    0    0    0    0    0    0    0    0    0   49  238  253  253  253
#> 
#> Columns 209 to 224 253  253  253  253  253  251   93   82   82   56   39    0    0    0    0    0
#> 
#> Columns 225 to 240   0    0    0    0    0    0    0   18  219  253  253  253  253  253  198  182
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,784} ]
```

We now define a more complex architecture that has one single input
which is a `lazy_tensor`. For that, we define first a single residual
layer:

``` r
layer = list(
  po("nop"),
  po("nn_linear", out_features = 50L) %>>%
    po("nn_dropout") %>>% po("nn_relu")
) %>>% po("nn_merge_sum")
layer$plot(horizontal = TRUE)
```

<img src="man/figures/README-unnamed-chunk-8-1.png" width="100%" />

We now define the input of the neural network to be a `lazy_tensor`
(`po("torch_ingress_num")`), apply a linear layer without output
dimension 50 and then repeat the above layer using the special
`PipeOpTorchBlock`, followed by the network’s head.and finally configure
the model arguments. After that, we configure the loss and the optimizer
and the training parameters.

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

Finally, we prepend the preprocessing step that flattens the images:

``` r
deep_learner = as_learner(
  flattener %>>% deep_network
)
deep_learner$id = "deep_network"
```

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
  easily tuned, `mlr3tuning` and friends

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
