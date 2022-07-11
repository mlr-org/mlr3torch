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
#' @template param_optimizer
#' @template param_loss
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerClassifAlexNet = R6Class("LearnerClassifAlexNet",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(optimizer = "adam", loss = "cross_entropy") {
      param_set = ps(
        pretrained = p_lgl(default = TRUE, tags = "train"),
        freeze = p_lgl(default = "TRUE iff pretrained", tags = "train",
          special_vals = list("TRUE iff pretrained")
        )
      )
      default_params = list(
        pretrained = TRUE
      )
      param_set$values = default_params

      super$initialize(
        id = "classif.alexnet",
        packages = c("torchvision", "torch"),
        param_set = param_set,
        feature_types = "imageuri",
        predict_types = "response",
        properties = c("multiclass", "twoclass"),
        man = "mlr3torch::mlr_learners_classif.alexnet",
        optimizer = optimizer,
        loss = loss,
        label = "AlexNet Image Classifier"
      )
    }
  ),
  private = list(
    .network = function(task) {
      p = self$param_set$get_values(tag = "train")
      pretrained = p$pretrained
      freeze = p$freeze %??% pretrained

      num_classes = nlevels(task$truth())
      if (pretrained) {
        network = torchvision::model_alexnet(pretrained = TRUE)
        network = reset_last_layer(
          model = network,
          num_classes = num_classes,
          bias = TRUE,
          freeze = freeze
        )
        return(network)
      }
      torchvision::model_alexnet(pretrained = FALSE, num_classes = num_classes)
    }
  )
)
