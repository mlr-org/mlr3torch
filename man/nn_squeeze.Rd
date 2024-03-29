% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PipeOpTorchReshape.R
\name{nn_squeeze}
\alias{nn_squeeze}
\title{Squeeze}
\usage{
nn_squeeze(dim)
}
\arguments{
\item{dim}{(\code{integer()})\cr
The dimension to squeeze.}
}
\description{
Squeezes a tensor by calling \code{\link[torch:torch_squeeze]{torch::torch_squeeze()}} with the given dimension \code{dim}.
}
