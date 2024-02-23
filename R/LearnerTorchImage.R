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
      callbacks = list(), packages = c("torchvision", "magick"), man, properties = NULL,
      predict_types = NULL) {
      properties = properties %??% switch(task_type,
        regr = c(),
        classif = c("twoclass", "multiclass")
      )
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
    .verify_train_task = function(task, param_vals) {
      if (!isTRUE(all.equal(task$feature_types$type, "lazy_tensor"))) {
        stopf("Must have exactly one feature of type lazy_tensor.")
      }
      assert_rgb_shape(c(
        c(NA, materialize(task$data(task$row_ids[1L], task$feature_names)[[1L]])[[1L]]$shape))
      )
      return(TRUE)
    },
    .dataset = function(task, param_vals) {
      dataset_ltnsr(task, param_vals)
    }

  )
)
