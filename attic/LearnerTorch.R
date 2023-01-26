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
#' the construction `param_set` and those from [`LearnerClassifTorchAbstract`].
#' @section Fields: `r roxy_fields(LearnerClassifTorch)`
#' @section Methods: `r roxy_methods(LearnerClassifTorch)`
#' @section Internals:
#'
#' @export
#' @include LearnerTorchAbstract.R
#' @examples
#' # TODO:
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(module, param_set = NULL, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      param_vals = list(), feature_types = NULL) {
      private$.module = module
      if (is.null(param_set)) {
        param_set = inferps(module)
        param_set$set_id = "net"
      } else {
        assert_true()
      }

      super$initialize(
        id = "classif.torch",
        properties = c("twoclass", "multiclass", "hotstart_forward"),
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
    .dataset = function(task, param_vals) {
      ingress_token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))
      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(num = ingress_token),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device %??% self$param_set$defaults$device
      )
    }
  )
)

#' @include zzz.R
mlr3torch_learners[["classif.torch"]] = LearnerClassifTorch
