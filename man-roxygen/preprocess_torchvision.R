#' <% pipeop = po(id) %>
#' @aliases <%= class(pipeop)[[1L]] %>
#' @usage NULL
#' @name mlr_pipeops_<%= id %>
#' @rdname mlr_pipeops_<%= id %>
#' @format [`R6Class`][R6::R6Class] inheriting from [`PipeOpTaskPreprocTorch`].
#' @section Construction:
#' ```r
#' po("<%= id%>"")
#' ````
#'
#' @description
#' Calls [`<%= paste0("torchvision::", gsub("^(augment|trafo)", "transform", id)) %>`],
#' see there for more information on the parameters.
#' <%= if (pipeop$rowwise) "The preprocessing is applied to each element of a batch individually." else "The preprocessing is applied to the whole batch."%>
#'
#' @section Parameters:
#' `r mlr3misc::rd_info(po("<%= id%>")$param_set)`
