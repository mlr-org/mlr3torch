#' @title LearnerTorchClassif
#'
#' @description
#' This implements a custom neural network.
#'
#' @name mlr_learners_classif.torch
#'
#' @template param_id
#' @template param_param_vals
#' @template param_optimizer
#' @template param_loss
#' @param feature_types (`character()`)\cr
#'   The feature types the learner supports. The default is all feature types.
#' @param network (`nn_module()`)\cr
#'   An object of class `"nn_module"` as defined in `torch`.
#'
#' @export
#' @examples
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(module, param_set = NULL, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), param_vals = list(), feature_types = NULL) {
      private$.module = module
      if (inherits(module, "nn_module_generator")) {
        param_set = inferps(module)
        param_set$set_id = "net"
      } else {
        stopf("Construction argument 'module' must either be NULL or a nn_module_generator.")
      }

      super$initialize(
        id = "classif.torch",
        properties = c("weights", "twoclass", "multiclass", "hotstart_forward"),
        label = "Classification Network",
        feature_types = feature_types %??% mlr_reflections$task_feature_types,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.torch",
        param_set = param_set
      )
    }
  ),
  private = list(
    .network = function(task) {
      pv = self$param_set$get_values(tags = "network")
      invoke(
        self$.module,
        task = task,
        .args = pv
      )
    },
    .module = NULL
  )
)
