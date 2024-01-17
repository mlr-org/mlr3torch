#' <% pipeop = po(id) %>
#' @title <%= class(pipeop)[[1L]] %>
#' @aliases <%= class(pipeop)[[1L]] %>
#' @usage NULL
#' @name mlr_pipeops_preproc_torch.<%= id %>
#' @rdname <%= class(pipeop)[[1L]] %>
#' @format [`R6Class`] inheriting from [`PipeOpTaskPreprocTorch`].
#'
#' @description
#' Calls [`<%= paste0("torchvision::",gsub("^(augment|trafo)", "transform", id)) %>`],
#' see there for more information on the parameters.
#' <%= if (pipeop$rowwise) "The preprocessing is applied row wise (no batch dimension)." else "The preprocessing is applied to the whole batch."%>
#'
#' @section Parameters:
#' `r mlr3misc::rd_info(po("<%= id%>")$param_set)`
