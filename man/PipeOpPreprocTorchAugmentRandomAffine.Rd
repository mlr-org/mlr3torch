% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/preprocess.R
\name{mlr_pipeops_augment_random_affine}
\alias{mlr_pipeops_augment_random_affine}
\alias{PipeOpPreprocTorchAugmentRandomAffine}
\title{PipeOpPreprocTorchAugmentRandomAffine}
\format{
\code{\link[R6:R6Class]{R6Class}} inheriting from \code{\link{PipeOpTaskPreprocTorch}}.
}
\description{
Calls \code{\link[torchvision:transform_random_affine]{torchvision::transform_random_affine}},
see there for more information on the parameters.
The preprocessing is applied row wise (no batch dimension).
}
\section{Parameters}{
\tabular{lllll}{
   Id \tab Type \tab Default \tab Levels \tab Range \cr
   degrees \tab untyped \tab - \tab  \tab - \cr
   translate \tab untyped \tab NULL \tab  \tab - \cr
   scale \tab untyped \tab NULL \tab  \tab - \cr
   resample \tab integer \tab 0 \tab  \tab \eqn{(-\infty, \infty)}{(-Inf, Inf)} \cr
   fillcolor \tab untyped \tab 0 \tab  \tab - \cr
   stages \tab character \tab - \tab train, predict, both \tab - \cr
   affect_columns \tab untyped \tab selector_all() \tab  \tab - \cr
}
}

