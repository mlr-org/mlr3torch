#' @title Tabular ResNet
#'
#' @name mlr_learners_classif.tab_resnet
#'
#' @description
#' Tabular resnet.
#'
#' @templateVar pkg torchvision
#' @templateVar model model_alexnet
#'
#' @template learner
#' @template pretrained
#'
#' @template param_optimizer
#' @template param_loss
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerClassifTabResNet = R6Class("LearnerClassifTabResNet",
  inherit = LearnerClassifTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(loss = "cross_entropy", optimizer = "adam") {
      param_set = make_paramset_tab_resnet_block()
      for (param in param_set$params_unid) {
        param$tags = unique(c(param$tags, "network"))
      }

      super$initialize(
        id = "classif.tab_resnet",
        packages = "torch",
        param_set = param_set,
        feature_types = c("numeric", "integer"),
        predict_types = c("response"),
        properties = c("multiclass", "twoclass"),
        man = "mlr3torch::mlr_learners_classif.tab_resnet",
        optimizer = optimizer,
        loss = loss,
        label = "Tabular ResNet"
      )

    }
  ),
  private = list(
    .network = function(task, param_vals) {
      pv = self$param_set$get_values(tags = "network")
      ii = startsWith(names(pv), "bn.")
      bn_args = pv[ii]
      names(bn_args) = gsub("bn.", "", names(bn_args))
      pv[ii] = NULL

      graph = top("input") %>>%
        top("select", items = "num") %>>%
        invoke(top, .obj = "tab_resnet_blocks", .args = pv) %>>%
        invoke(top, .obj = "batch_norm", .args = bn_args) %>>%
        invoke(top, .obj = pv$activation, .args = pv$activation_args) %>>%
        top("output")

      network = graph$train(task)[[1L]]$network

      return(network)
    }
  )
)
