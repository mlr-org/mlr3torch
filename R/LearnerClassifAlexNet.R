#' @title AlexNet Image Classifier
#'
#' @name mlr_learners_classif.alexnet
#'
#' @description
#' Convolutional network for image classification.
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
#' `r format_bib("krizhevsky2014one")`
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerClassifAlexNet = R6Class("LearnerClassifAlexNet",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_.optimizer
    #' @template param_.loss
    initialize = function(.optimizer = "adam", .loss = "cross_entropy") {
      param_set = ps(
        pretrained = p_lgl(default = TRUE, tags = "train")
      )
      param_set$values$pretrained = TRUE

      super$initialize(
        id = "classif.alexnet",
        packages = c("torchvision", "torch"),
        param_set = param_set,
        feature_types = "imageuri",
        predict_types = "response",
        properties = c("multiclass", "twoclass"),
        man = "mlr3torch::mlr_learners_classif.alexnet",
        optimizer = .optimizer,
        loss = .loss,
        label = "AlexNet Image Classifier"
      )
    }
  ),
  private = list(
    .network = function(task) {
      pretrained = self$param_set$values$pretrained
      num_classes = nlevels(task$truth())
      if (pretrained) {
        network = torchvision::model_alexnet(pretrained = TRUE)
        for (par in network$parameters) {
          par$requires_grad_(FALSE)
        }
        network = reset_last_layer(network, num_classes)
        return(network)
      }
      torchvision::model_alexnet(pretrained = FALSE, num_classes = num_classes)
    }
  )
)
