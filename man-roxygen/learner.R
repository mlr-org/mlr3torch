#' <% task_types_vec <- strsplit(task_types, split = ", ")[[1L]]%>
#'
#' @name <%=paste0("mlr_learners.", name)%>
#'
#' @include LearnerTorch.R
#'
#' @section Dictionary:
#' This [Learner][mlr3::Learner] can be instantiated using the sugar function [`lrn()`][mlr3::lrn]:
#' ```
#' <%= if ("classif" %in% task_types_vec) paste0("lrn(\"classif.", name, "\", ...)") else ""%>
#' <%= if ("regr" %in% task_types_vec) paste0("lrn(\"regr.", name, "\", ...)") else ""%>
#' ```
#'
#' @section Properties:
#' `r mlr3torch:::rd_info_learner_torch("<%=name%>", "<%=task_types%>")`
#' @md
#'
#' @family Learner
