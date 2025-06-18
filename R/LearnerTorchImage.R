#' @title Image Learner
#'
#' @name mlr_learners_torch_image
#'
#' @description
#' Base Class for Image Learners.
#' The features are assumed to be a single [`lazy_tensor`] column in RGB format.
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
#' @param jittable (`logical(1)`)\cr
#'   Whether the model can be jit-traced.
#'
#' @section Parameters:
#' Parameters include those inherited from [`LearnerTorch`] and the `param_set` construction argument.
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
    initialize = function(id, task_type, param_set = ps(), label, optimizer = NULL, loss = NULL,
      callbacks = list(), packages, man, properties = NULL, predict_types = NULL, jittable = FALSE) {
      super$initialize(
        id = id,
        task_type = task_type,
        label = label,
        optimizer = optimizer,
        loss = loss,
        param_set = param_set,
        packages = packages,
        callbacks = callbacks,
        predict_types = predict_types,
        feature_types = "lazy_tensor",
        man = man,
        jittable = jittable
      )
    }
  ),
  private = list(
    .ingress_tokens = function(task, param_vals) {
      if (task$n_features != 1L) {
        stopf("Learner '%s' received task '%s' with %i features, but the learner expects exactly one feature.", self$id, task$id, length(task$feature_names))
      }
      list(input = ingress_ltnsr(feature_name = task$feature_names))
    }
  )
)
