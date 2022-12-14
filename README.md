
<!-- README.md is generated from README.Rmd. Please edit that file -->

# mlr3torch

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![r-cmd-check](https://github.com/mlr-org/mlr3torch/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/mlr-org/mlr3torch/actions/workflows/r-cmd-check.yml)
[![CRAN
status](https://www.r-pkg.org/badges/version/mlr3torch)](https://CRAN.R-project.org/package=mlr3torch)
[![StackOverflow](https://img.shields.io/badge/stackoverflow-mlr3-orange.svg)](https://stackoverflow.com/questions/tagged/mlr3)
[![Mattermost](https://img.shields.io/badge/chat-mattermost-orange.svg)](https://lmmisld-lmu-stats-slds.srv.mwn.de/mlr_invite/)
<!-- badges: end -->

The goal of {mlr3torch} is to connect
[{mlr3}](https://github.com/mlr-org/mlr3) with
[{torch}](https://github.com/mlverse/torch).

It is in the very early stages of development and it’s future and scope
are yet to be determined.

## Installation

``` r
remotes::install_github("mlr-org/mlr3torch")
```

## `tabnet` Example

Using the [{tabnet}](https://github.com/mlverse/tabnet) learner for
classification:

``` r
library(mlr3)
library(mlr3viz)
library(mlr3torch)

task = tsk("german_credit")

# Set up the learner
lrn_tabnet = lrn("classif.tabnet", epochs = 5)

# Train and Predict
lrn_tabnet$train(task, row_ids = 1:900)

preds = lrn_tabnet$predict(task, row_ids = 901:1000)

# Investigate predictions
preds$confusion
preds$score(msr("classif.acc"))

# Predict probabilities instead
lrn_tabnet$predict_type = "prob"
preds_prob = lrn_tabnet$predict(task)

autoplot(preds_prob, type = "roc")

# Examine variable importance scores
lrn_tabnet$importance()
```

## Using `TorchOp`s

``` r
task = tsk("iris")

graph = top("input") %>>%
  top("tokenizer_tabular", d_token = 1) %>>%
  top("flatten") %>>%
  top("relu_1") %>>%
  top("linear_1", out_features = 10) %>>%
  top("relu_2") %>>%
  top("head") %>>%
  top("model.classif", epochs = 10L, batch_size = 16L, .loss = "cross_entropy", .optimizer = "adam")

glrn = as_learner_torch(graph)
glrn$train(task)
```

## Credit

Some parts of the implementation are inspired by other deep learning
libraries:

  - [Keras](https://keras.io/) - Building networks using `TorchOp`’s
    feels similar to using `keras`.
  - [Luz](https://github.com/mlverse/luz) - Our implementation of
    callbacks is inspired by the R package `luz`
