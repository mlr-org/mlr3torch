% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CallbackSet.R
\name{callback_set}
\alias{callback_set}
\title{Create a Set of Callbacks for Torch}
\usage{
callback_set(
  classname,
  on_begin = NULL,
  on_end = NULL,
  on_exit = NULL,
  on_epoch_begin = NULL,
  on_before_valid = NULL,
  on_epoch_end = NULL,
  on_batch_begin = NULL,
  on_batch_end = NULL,
  on_after_backward = NULL,
  on_batch_valid_begin = NULL,
  on_batch_valid_end = NULL,
  on_valid_end = NULL,
  state_dict = NULL,
  load_state_dict = NULL,
  initialize = NULL,
  public = NULL,
  private = NULL,
  active = NULL,
  parent_env = parent.frame(),
  inherit = CallbackSet,
  lock_objects = FALSE
)
}
\arguments{
\item{classname}{(\code{character(1)})\cr
The class name.}

\item{on_begin, on_end, on_epoch_begin, on_before_valid, on_epoch_end, on_batch_begin, on_batch_end, on_after_backward, on_batch_valid_begin, on_batch_valid_end, on_valid_end, on_exit}{(\code{function})\cr
Function to execute at the given stage, see section \emph{Stages}.}

\item{state_dict}{(\verb{function()})\cr
The function that retrieves the state dict from the callback.
This is what will be available in the learner after training.}

\item{load_state_dict}{(\verb{function(state_dict)})\cr
Function that loads a callback state.}

\item{initialize}{(\verb{function()})\cr
The initialization method of the callback.}

\item{public, private, active}{(\code{list()})\cr
Additional public, private, and active fields to add to the callback.}

\item{parent_env}{(\code{environment()})\cr
The parent environment for the \code{\link[R6:R6Class]{R6Class}}.}

\item{inherit}{(\code{R6ClassGenerator})\cr
From which class to inherit.
This class must either be \code{\link{CallbackSet}} (default) or inherit from it.}

\item{lock_objects}{(\code{logical(1)})\cr
Whether to lock the objects of the resulting \code{\link[R6:R6Class]{R6Class}}.
If \code{FALSE} (default), values can be freely assigned to \code{self} without declaring them in the
class definition.}
}
\value{
\code{\link{CallbackSet}}
}
\description{
Creates an \code{R6ClassGenerator} inheriting from \code{\link{CallbackSet}}.
Additionally performs checks such as that the stages are not accidentally misspelled.
To create a \code{\link{TorchCallback}} use \code{\link[=torch_callback]{torch_callback()}}.

In order for the resulting class to be cloneable, the private method \verb{$deep_clone()} must be
provided.
}
\seealso{
Other Callback: 
\code{\link{TorchCallback}},
\code{\link{as_torch_callback}()},
\code{\link{as_torch_callbacks}()},
\code{\link{mlr3torch_callbacks}},
\code{\link{mlr_callback_set}},
\code{\link{mlr_callback_set.checkpoint}},
\code{\link{mlr_callback_set.progress}},
\code{\link{mlr_callback_set.tb}},
\code{\link{mlr_callback_set.unfreeze}},
\code{\link{mlr_context_torch}},
\code{\link{t_clbk}()},
\code{\link{torch_callback}()}
}
\concept{Callback}
