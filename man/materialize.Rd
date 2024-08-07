% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/materialize.R
\name{materialize}
\alias{materialize}
\alias{materialize.list}
\title{Materialize Lazy Tensor Columns}
\usage{
materialize(x, device = "cpu", rbind = FALSE, ...)

\method{materialize}{list}(x, device = "cpu", rbind = FALSE, cache = "auto", ...)
}
\arguments{
\item{x}{(any)\cr
The object to materialize.
Either a \code{\link{lazy_tensor}} or a \code{list()} / \code{data.frame()} containing \code{\link{lazy_tensor}} columns.}

\item{device}{(\code{character(1)})\cr
The torch device.}

\item{rbind}{(\code{logical(1)})\cr
Whether to rbind the lazy tensor columns (\code{TRUE}) or return them as a list of tensors (\code{FALSE}).
In the second case, there is no batch dimension.}

\item{...}{(any)\cr
Additional arguments.}

\item{cache}{(\code{character(1)} or \code{environment()} or \code{NULL})\cr
Optional cache for (intermediate) materialization results.
Per default, caching will be enabled when the same dataset or data descriptor (with different output pointer)
is used for more than one lazy tensor column.}
}
\value{
(\code{list()} of \code{\link{lazy_tensor}}s or a \code{\link{lazy_tensor}})
}
\description{
This will materialize a \code{\link[=lazy_tensor]{lazy_tensor()}} or a \code{data.frame()} / \code{list()} containing -- among other things --
\code{\link[=lazy_tensor]{lazy_tensor()}} columns.
I.e. the data described in the underlying \code{\link{DataDescriptor}}s is loaded for the indices in the \code{\link[=lazy_tensor]{lazy_tensor()}},
is preprocessed and then put unto the specified device.
Because not all elements in a lazy tensor must have the same shape, a list of tensors is returned by default.
If all elements have the same shape, these tensors can also be rbinded into a single tensor (parameter \code{rbind}).
}
\details{
Materializing a lazy tensor consists of:
\enumerate{
\item Loading the data from the internal dataset of the \code{\link{DataDescriptor}}.
\item Processing these batches in the preprocessing \code{\link[mlr3pipelines:Graph]{Graph}}s.
\item Returning the result of the \code{\link[mlr3pipelines:PipeOp]{PipeOp}} pointed to by the \code{\link{DataDescriptor}} (\code{pointer}).
}

With multiple \code{\link{lazy_tensor}} columns we can benefit from caching because:
a) Output(s) from the dataset might be input to multiple graphs.
b) Different lazy tensors might be outputs from the same graph.

For this reason it is possible to provide a cache environment.
The hash key for a) is the hash of the indices and the dataset.
The hash key for b) is the hash of the indices, dataset and preprocessing graph.
}
\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
lt1 = as_lazy_tensor(torch_randn(10, 3))
materialize(lt1, rbind = TRUE)
materialize(lt1, rbind = FALSE)
lt2 = as_lazy_tensor(torch_randn(10, 4))
d = data.table::data.table(lt1 = lt1, lt2 = lt2)
materialize(d, rbind = TRUE)
materialize(d, rbind = FALSE)
\dontshow{\}) # examplesIf}
}
