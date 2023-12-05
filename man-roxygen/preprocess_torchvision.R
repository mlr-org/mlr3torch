#' <% pipeop = po(id) %>
#' @title <%= class(pipeop)[[1L]] %>
#' @usage NULL
#' @name mlr_pipeops_preproc_torch.<%= id %>
#' @rdname <%= class(pipeop)[[1L]] %>
#' @format [`R6Class`] inheriting from [`PipeOpTaskPreprocTorch`].
#'
#' @description
#' Calls [`<%= paste0("torchvision::",gsub("^(augment|trafo)", "transform", id)) %>`],
#' see there for more information on the parameters.
#'
#' @section Parameters:
#' `r mlr3misc::rd_info(po("<%= id%>")$param_set)`
