#' @title Classification Torch Learner
#'
#' @usage  NULL
#' @name mlr_learners_classif.torch
#' @format `r roxy_format(LearnerClassifTorch)`
#'
#' @description
#' Custom torch classification network.
#'
#' @section Construction: `r roxy_construction(LearnerClassifTorch)`
#' * `module` ::
#'   An object of class `"nn_module"` as defined in `torch`.
#'   The output is expected to be the scores, i.e. the output before the final softmax layer.
#' *  `param_set` ::
#' * `optimizer` ::
#' * `loss` ::
#' * `param_vals` ::
#' * `feature_types` ::
#'   The feature types the learner supports. The default is all feature types.
#'
#' @section State: See [`LearnerClassifTorchAbstact`].
#' @section Parameters: 
#' The union of: 
#' * The construction `param_set` (is inferred if it is not s)
#' the construction `param_set` and those from [`LearnerClassifTorch`].
#' @section Fields: `r roxy_fields_inherit(LearnerClassifTorch)`
#' @section Methods: `r roxy_methods_inherit(LearnerClassifTorch)`
#' @section Internals:
#'
#' @export
#' @include LearnerTorch.R
#' @examples
#' # TODO:
LearnerClassifTorchModule = R6Class("LearnerClassifTorchModule",
  inherit = LearnerClassifTorch,
  public = list(
    initialize = function(module, param_set = NULL, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      param_vals = list(), feature_types = NULL, dataset) {
      private$.module = module
      private$.dataset = assert_function(dataset, args = c("task", "param_vals"))
      if (is.null(param_set)) {
        param_set = inferps(module)
        param_set$set_id = "net"
      } else {
        assert_true(TRUE)
      }

      super$initialize(
        id = "classif.torch_module",
        properties = c("twoclass", "multiclass"),
        label = "Torch Module Classifier",
        feature_types = feature_types %??% mlr_reflections$task_feature_types,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.torch",
        param_set = param_set
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      invoke(
        private$.module,
        task = task,
        .args = param_vals
      )
    },
    .module = NULL,
    .dataset = NULL
  )
)

#' @include zzz.R
register_learner("classif.torch_module", LearnerClassifTorchModule)
