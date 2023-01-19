#' @title AlexNet Image Classifier
#' @usage NULL
#' @format [`R6Class`] inheriting from [`LearnerClassifTorchImage`].
#' @name mlr_learners_classif.alexnet
#'
#' @description
#' Convolutional network for image classification.
#'
#' @section Parameters:
#' Parameters from [`LearnerClassifTorchImage`] and:
#'
#' * `pretrained` :: `logical(1)`/cd
#'   Whether to use the pretrained model.
#'
#' @include LearnerClassifTorchImage.R
#' @export
LearnerClassifAlexNet = R6Class("LearnerClassifAlexNet",
  inherit = LearnerClassifTorchImage,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_opt("cross_entropy")) {
      param_set = ps(
        pretrained = p_lgl(default = TRUE, tags = "train"),
      )

      super$initialize(
        id = "classif.alexnet",
        param_set = param_set,
        predict_types = "response",
        man = "mlr3torch::mlr_learners_classif.alexnet",
        optimizer = optimizer,
        loss = loss,
        label = "AlexNet Image Classifier"
      )
    }
  ),
  private = list(
    .network = function(task) {
      param_vals = self$param_set$get_values()
      param_vals$pretrained = param_vals$pretrained %??% TRUE

      if (pretrained) {
        network = torchvision::model_alexnet(pretrained = param_vals$pretrained)

        network$classifier$`6` = torch::nn_linear(
          in_features = model$classifier$`6`$in_features,
          out_features = length(task$target_names),
          bias = param_values$bias
        )
        return(network)
      }

      torchvision::model_alexnet(pretrained = FALSE, num_classes = length(task$target_names))
    }
  )
)

#' @include aaa.R
mlr3torch_learners$classif.alexnet = LearnerClassifAlexNet
