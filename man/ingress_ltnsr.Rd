% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PipeOpTorchIngress.R
\name{ingress_ltnsr}
\alias{ingress_ltnsr}
\title{Ingress Token for Lazy Tensor Feature}
\usage{
ingress_ltnsr(feature_name = NULL)
}
\arguments{
\item{feature_name}{(\code{character(1)})\cr
Which lazy tensor feature to select if there is more than one.}
}
\value{
\code{\link{TorchIngressToken}}
}
\description{
Represents an entry point representing a tensor containing a single lazy tensor feature.
}
