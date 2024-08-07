% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/TaskClassif_lazy_iris.R
\name{mlr_tasks_lazy_iris}
\alias{mlr_tasks_lazy_iris}
\title{Iris Classification Task}
\format{
\link[R6:R6Class]{R6::R6Class} inheriting from \link[mlr3:TaskClassif]{mlr3::TaskClassif}.
}
\source{
\url{https://en.wikipedia.org/wiki/Iris_flower_data_set}
}
\description{
A classification task for the popular \link[datasets:iris]{datasets::iris} data set.
Just like the iris task, but the features are represented as one lazy tensor column.
}
\section{Construction}{


\if{html}{\out{<div class="sourceCode">}}\preformatted{tsk("lazy_iris")
}\if{html}{\out{</div>}}
}

\section{Properties}{

\itemize{
\item Task type: \dQuote{classif}
\item Properties: \dQuote{multiclass}
\item Has Missings: no
\item Target: \dQuote{Species}
\item Features: \dQuote{x}
\item Data Dimension: 150x3
}
}

\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
task = tsk("lazy_iris")
task
df = task$data()
materialize(df$x[1:6], rbind = TRUE)
\dontshow{\}) # examplesIf}
}
\references{
Anderson E (1936).
\dQuote{The Species Problem in Iris.}
\emph{Annals of the Missouri Botanical Garden}, \bold{23}(3), 457.
\doi{10.2307/2394164}.
}
