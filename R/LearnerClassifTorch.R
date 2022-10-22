#' @title LearnerTorchClassif
#'
#' @usage  NULL
#' @name mlr_learners_classif.torch
#' @format [`R6Class`] inheriting from [`LearnerClassifTorchAbstract`].
#'
#' @description
#' This implements a custom neural network.
#'
#' @section Construction:
#' ```
#' LearnerClassifTorch$new(module, param_set = NULL, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
#'   param_vals = list(), feature_types = NULL)
#' ```
#' * `module` ::
#'   An object of class `"nn_module"` as defined in `torch`.
#' *  `param_set` ::
#' * `optimizer` ::
#' * `loss` ::
#' * `param_vals` ::
#' * `feature_types` ::
#'   The feature types the learner supports. The default is all feature types.
#'
#' @section Internals:
#' TODO:
#'
#' @section Parameters:
#' The parameter set defined as argument `param_set` during construction.
#'
#' @section Fields:
#' Only fields inherited from [`LearnerClassifTorch`], [`LearnerClassif`] or [`Learner`].
#'
#' @section Methods:
#' Only methods inherited from [`LearnerClassifTorch`], [`LearnerClassif`] or [`Learner`].
#'
#' @export
#' @examples
#' # TODO:
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(module, param_set = NULL, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      param_vals = list(), feature_types = NULL) {
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
