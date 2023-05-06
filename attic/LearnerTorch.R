#' @title Classification Torch Learner
#'
#' @name mlr_learners_classif.torch
#'
#' @description
#' Custom torch classification network.
#'
#'
#' @section Parameters:
#' The union of:
#' * The construction `param_set` (is inferred if it is not s)
#' the construction `param_set` and those from [`LearnerClassifTorch`].
#' @section Internals:
#' TODO
#'
#' @export
#' @include LearnerTorch.R
#' @examples
#' # TODO:
LearnerClassifTorchModule = R6Class("LearnerClassifTorchModule",
  inherit = LearnerClassifTorch,
  public = list(
    #' @description
    initialize = function(module, param_set = NULL, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      param_vals = list(), feature_types = NULL, dataset) {
      private$.module = module
      private$.dataset = assert_function(dataset, args = c("task", "param_vals"))
      if (is.null(param_set)) {
        param_set = inferps(module)
        param_set$set_id = "net"
      } else {
      }
      assert_true(FALSE)

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
