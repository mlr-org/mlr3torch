#' @title Image Classification Network
#'
#' @name mlr_learners_classif_torch_image
#'
#' @description
#' Base Class for Image Classification Learners.
#'
#' @template param_id
#' @template param_param_set
#' @template param_optimizer
#' @template param_callbacks
#' @template param_loss
#' @template param_packages
#' @template param_man
#' @template param_properties
#' @template param_label
#'
#' @section Inheriting:
#' To inherit from this class, one should overwrite the private `$.network()` to return a [`nn_module`] that has
#' one argument in its forward method.
#'
#' @section Parameters:
#' Parameters include those inherited from [`LearnerClassifTorch`], the `param_set` construction argument, as
#' well as:
#'
#' * `channels` :: `integer(1)` \cr
#'   The number of input channels.
#' * `height` :: `integer(1)` \cr
#'   The height of the input image.
#' * `width` :: `integer(1)` \cr
#'   The width of the input image.
#'
#' @family Learner
#' @include LearnerTorch.R
#'
#' @export
LearnerClassifTorchImage = R6Class("LearnerClassifTorchImage",
  inherit = LearnerClassifTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, param_set, label, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      callbacks = list(), packages = c("torchvision", "magick"), man, properties = c("twoclass", "multiclass")) {
      assert_param_set(param_set)
      predefined_set = ps(
        channels   = p_int(1, tags = c("train", "predict", "required")),
        height     = p_int(1, tags = c("train", "predict", "required")),
        width      = p_int(1, tags = c("train", "predict", "required"))
      )

      if (param_set$length) {
        param_set$add(predefined_set)
      } else {
        param_set = predefined_set
      }

      super$initialize(
        id = id,
        label = label,
        optimizer = optimizer,
        properties = properties,
        loss = loss,
        param_set = param_set,
        packages = packages,
        callbacks = callbacks,
        predict_types = c("response", "prob"),
        feature_types = "imageuri",
        man = man
      )
    }
  ),
  private = list(
    .dataset = function(task, param_vals) {
      dataset_img(task, param_vals)
    }
  )
)
