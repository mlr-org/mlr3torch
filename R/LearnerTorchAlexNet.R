# TODO
#' @title AlexNet Image Classifier
#'
#' @templateVar name alexnet
#' @templateVar task_types classif
#' @template learner
#' @template params_learner
#'
#' @description
#' Historic convolutional neural network for image classification.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorchImage`] and
#'
#' * `pretrained` :: `logical(1)`\cr
#'   Whether to use the pretrained model.
#'
#' @references `r format_bib("krizhevsky2017imagenet")`
#' @include LearnerTorchImage.R
#' @export
LearnerTorchAlexNet = R6Class("LearnerTorchAlexNet",
  inherit = LearnerTorchImage,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      param_set = ps(
        pretrained = p_lgl(default = TRUE, tags = "train")
      )
      # TODO: Freezing --> maybe as a callback?
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".alexnet"),
        param_set = param_set,
        man = "mlr3torch::mlr_learners.alexnet",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        label = "AlexNet Image Classifier"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nout = if (self$task_type == "regr") 1 else length(task$class_names)
      if (param_vals$pretrained %??% TRUE) {
        network = torchvision::model_alexnet(pretrained = TRUE)

        network$classifier$`6` = torch::nn_linear(
          in_features = network$classifier$`6`$in_features,
          out_features = nout,
          bias = TRUE
        )
        return(network)
      }

      torchvision::model_alexnet(pretrained = FALSE, num_classes = nout)
    }
  )
)

#' @include zzz.R
register_learner("classif.alexnet", LearnerTorchAlexNet)
