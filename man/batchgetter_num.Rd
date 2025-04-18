% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/task_dataset.R
\name{batchgetter_num}
\alias{batchgetter_num}
\title{Batchgetter for Numeric Data}
\usage{
batchgetter_num(data, ...)
}
\arguments{
\item{data}{(\code{data.table()})\cr
\code{data.table} to be converted to a \code{tensor}.}

\item{...}{(any)\cr
Unused.}
}
\description{
Converts a data frame of numeric data into a float tensor by calling \code{as.matrix()}.
No input checks are performed
}
