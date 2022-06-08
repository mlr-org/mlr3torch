#' @title Tabular ResNet
#'
#' @name mlr_learners_classif.tab_resnet
#'
#' @description
#' Tabular resnet.
#'
#' @templateVar id classif.alexnet
#' @templateVar pkg torchvision
#' @templateVar model model_alexnet
#'
#' @template learner
#' @template pretrained
#' @template optimizer
#' @template loss_classif
#'
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerClassifTabResNet = R6Class("LearnerClassifTabResNet",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_.optimizer
    #' @template param_.loss
    initialize = function(.optimizer = "adam", .loss = "cross_entropy") {
      private$.block = top("tab_resnet.block", id = "")

      param_set = private$.block$param_set

      super$initialize(
        id = "classif.tab_resnet",
        packages = c("torchvision", "torch"),
        param_set = param_set,
        feature_types = c("numeric", "integer"),
        predict_types = "response",
        properties = c("multiclass", "twoclass"),
        man = "mlr3torch::mlr_learners_classif.tab_resnet",
        .optimizer = .optimizer,
        .loss = .loss,
        label = "Tabular ResNet"
      )
    }
  ),
  private = list(
    .network = function(task) {
      build_tabular_resnet(self, private$.block, task)
    },
    .block = NULL
  )
)

