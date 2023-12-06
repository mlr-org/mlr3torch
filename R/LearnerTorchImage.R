#' @title Image Network
#'
#' @name mlr_learners_torch_image
#'
#' @description
#' Base Class for Torch Image Learners.
#'
#' @template param_id
#' @template param_task_type
#' @template param_param_set
#' @template param_optimizer
#' @template param_callbacks
#' @template param_loss
#' @template param_packages
#' @template param_man
#' @template param_properties
#' @template param_label
#' @template param_predict_types
#'
#' @section State:
#' The state is a list with elements `network`, `optimizer`, `loss_fn`, `callbacks` and `seed`.
#'
#' @section Inheriting:
#' To inherit from this class, one should overwrite the private `$.network()` method to return a
#' [`nn_module`] that has one argument in its forward method.
#'
#' @section Parameters:
#' Parameters include those inherited from [`LearnerTorch`], the `param_set` construction argument, as
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
LearnerTorchImage = R6Class("LearnerTorchImage",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, task_type, param_set, label, optimizer = NULL, loss = NULL,
      callbacks = list(), packages = c("torchvision", "magick"), man, properties = NULL,
      predict_types = NULL) {
      properties = properties %??% switch(task_type,
        regr = character(0),
        classif = c("twoclass", "multiclass")
      )
      assert_param_set(param_set)

      super$initialize(
        id = id,
        task_type = task_type,
        label = label,
        optimizer = optimizer,
        properties = properties,
        loss = loss,
        param_set = param_set,
        packages = packages,
        callbacks = callbacks,
        predict_types = predict_types,
        feature_types = "lazy_tensor",
        man = man
      )
    }
  ),
  private = list(
    .verify_train_task = function(task, row_ids) {
      if (length(task$feature_names) != 1L || !identical(task$feature_types$type, "lazy_tensor")) {
        stopf("Must have exactly one feature of type lazy_tensor.")
      }
    },
    .dataset = function(task, param_vals) {
      dataset_ltnsr(task, param_vals)
    }
  )
)
