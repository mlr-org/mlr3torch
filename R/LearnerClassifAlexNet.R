#' @title AlexNet Image Classifier
#'
#' @usage NULL
#' @name mlr_learners_classif.alexnet
#' @format `r roxy_format(LearnerClassifAlexNet)`
#'
#' @description
#' Convolutional network for image classification.
#'
#' @section Parameters:
#' Parameters from [`LearnerClassifTorchImage`] and
#'
#' * `pretrained` :: `logical(1)`\cr
#'   Whether to use the pretrained model.
#'
#'  TODO: Finish description
#'
#' @include LearnerClassifTorchImage.R
#' @export
LearnerClassifAlexNet = R6Class("LearnerClassifAlexNet",
  inherit = LearnerClassifTorchImage,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), callbacks = list()) {
      param_set = ps(
        pretrained = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = "classif.alexnet",
        param_set = param_set,
        man = "mlr3torch::mlr_learners_classif.alexnet",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        label = "AlexNet Image Classifier"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      if (param_vals$pretrained %??% TRUE) {
        network = torchvision::model_alexnet(pretrained = TRUE)

        network$classifier$`6` = torch::nn_linear(
          in_features = network$classifier$`6`$in_features,
          out_features = length(task$class_names),
          bias = TRUE
        )
        return(network)
      }

      torchvision::model_alexnet(pretrained = FALSE, num_classes = length(task$class_names))
    }
  )
)

#' @include zzz.R
register_learner("classif.alexnet", LearnerClassifAlexNet)
