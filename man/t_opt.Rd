% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/TorchOptimizer.R
\name{t_opt}
\alias{t_opt}
\alias{t_opts}
\title{Optimizers Quick Access}
\usage{
t_opt(.key, ...)

t_opts(.keys, ...)
}
\arguments{
\item{.key}{(\code{character(1)})\cr
Key of the object to retrieve.}

\item{...}{(any)\cr
See description of \code{\link[mlr3misc:dictionary_sugar_get]{dictionary_sugar_get}}.}

\item{.keys}{(\code{character()})\cr
The keys of the optimizers.}
}
\value{
A \code{\link{TorchOptimizer}}
}
\description{
Retrieves one or more \code{\link{TorchOptimizer}}s from \code{\link{mlr3torch_optimizers}}.
Works like \code{\link[mlr3:mlr_sugar]{mlr3::lrn()}} and \code{\link[mlr3:mlr_sugar]{mlr3::lrns()}}.
}
\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
t_opt("adam", lr = 0.1)
# get the dictionary
t_opt()
\dontshow{\}) # examplesIf}
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
t_opts(c("adam", "sgd"))
# get the dictionary
t_opts()
\dontshow{\}) # examplesIf}
}
\seealso{
Other Torch Descriptor: 
\code{\link{TorchCallback}},
\code{\link{TorchDescriptor}},
\code{\link{TorchLoss}},
\code{\link{TorchOptimizer}},
\code{\link{as_torch_callbacks}()},
\code{\link{as_torch_loss}()},
\code{\link{as_torch_optimizer}()},
\code{\link{mlr3torch_losses}},
\code{\link{mlr3torch_optimizers}},
\code{\link{t_clbk}()},
\code{\link{t_loss}()}

Other Dictionary: 
\code{\link{mlr3torch_callbacks}},
\code{\link{mlr3torch_losses}},
\code{\link{mlr3torch_optimizers}}
}
\concept{Dictionary}
\concept{Torch Descriptor}
