#' @name <%=paste0("mlr_learners_", id)%>
#'
#' @include LearnerTorch.R
#'
#' @section Dictionary:
#' This [Learner] can be instantiated via the [dictionary][mlr3misc::Dictionary] [mlr_learners] or with the associated sugar function [lrn()]:
#' ```
#' mlr_learners$get("<%= id %>")
#' lrn("<%= id %>")
#' ```
#'
#' @section Meta Information:
#' `r mlr3misc::rd_info(mlr3::lrn("<%= id %>"))`
#' @md
