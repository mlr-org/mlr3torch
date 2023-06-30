#' <% task_types_vec <- strsplit(task_types, split = ", ")[[1L]]%>
#'
#' @name <%=paste0("mlr_learners.", name)%>
#'
#' @include LearnerTorch.R
#'
#' @section Dictionary:
#' This [Learner] can be instantiated using the sugar function [lrn()]:
#' ```
#' <%= if ("classif" %in% task_types_vec) paste0("lrn(\"classif.", name, "\", ...)") else ""%>
#' <%= if ("regr" %in% task_types_vec) paste0("lrn(\"regr.", name, "\", ...)") else ""%>
#' ```
#'
#' @section Meta Information:
#' `r mlr3torch:::rd_info_learner_torch("<%=name%>", "<%=task_types%>")`
#' @md
#'
#' @section State:
#' The state is a list with elements:
#'   * `network` :: The trained [network][torch::nn_module].
#'   * `optimizer` :: The [optimizer][torch::optimizer] used to train the network.
#'   * `loss_fn` :: The [loss][torch::nn_module] used to train the network.
#'   * `callbacks` :: The [callbacks][mlr3torch::mlr_callbacks_torch] used to train the network.
#'   * `seed` :: The actual seed that was / is used for training and prediction.
#'
#' @family Learner
