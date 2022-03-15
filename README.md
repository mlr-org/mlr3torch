
<!-- README.md is generated from README.Rmd. Please edit that file -->

# mlr3torch

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/mlr-org/mlr3torch/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mlr-org/mlr3torch/actions/workflows/R-CMD-check.yaml)
[![CRAN
status](https://www.r-pkg.org/badges/version/mlr3torch)](https://CRAN.R-project.org/package=mlr3torch)
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


# Credit
This API is heavily inspired by:

* Keras

``` r
library(mlr3)
library(mlr3viz)
library(mlr3torch)

task <- tsk("german_credit")

# Set up the learner
lrn_tabnet <- lrn("classif.torch.tabnet", epochs = 5)

# Train and Predict
lrn_tabnet$train(task, row_ids = 1:900)

preds <- lrn_tabnet$predict(task, row_ids = 901:1000)

# Investigate predictions
preds$confusion
preds$score(msr("classif.acc"))

# Predict probabilities instead
lrn_tabnet$predict_type <- "prob"
preds_prob <- lrn_tabnet$predict(task)

autoplot(preds_prob, type = "roc")

# Examine variable importance scores
lrn_tabnet$importance()
```
