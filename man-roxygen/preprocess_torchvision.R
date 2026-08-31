#' <% pipeop = po(id) %>
#' @aliases <%= class(pipeop)[[1L]] %>
#' @usage NULL
#' @name mlr_pipeops_<%= id %>
#' @rdname mlr_pipeops_<%= id %>
#' @format [`R6Class`][R6::R6Class] inheriting from [`PipeOpTaskPreprocTorch`].
#' @section Construction:
#' ```r
#' po("<%= id%>")
#' ```
#'
#' @description
#' Calls [`<%= paste0("torchvision::", gsub("^(augment|trafo)", "transform", id)) %>`],
#' see there for more information on the parameters.
#' <%= if (pipeop$rowwise) "The preprocessing is applied to each element of a batch individually." else "The preprocessing is applied to the whole batch."%>
#'
#' <%= if (grepl("^augment", id)) "Being an `augment_` operator, its `stages` parameter starts out as `\"train\"`, so the augmentation is applied while training and not when predicting. Set `stages = \"both\"` to apply it in both phases." else "Being a `trafo_` operator, its `stages` parameter starts out as `\"both\"`, so the transformation is applied both while training and when predicting. Set `stages = \"train\"` to restrict it to training." %>
#' See [`PipeOpTaskPreprocTorch`] for the `stages` parameter and this naming convention.
#'
#' @section Parameters:
#' `r mlr3torch:::rd_info_param_set(po("<%= id%>")$param_set)`
