#' @title Classification Torch Learner
#'
#' @usage NULL
#' @name mlr_learners_classif.torch_model
#' @format `r roxy_format(LearnerClassifTorchModel)`
#'
#' @description
#' Classification Torch Learner that is used internally by [`PipeOpTorchModelClassif`].
#'
#' @section Construction: `r roxy_construction(LearnerClassifTorchModel)`
#' * `network` :: [`nn_module`]\cr
#' * `ingress_tokens` :: `list`\cr
#' 
#'
#' @section State: See [`LearnerClassifTorch`]
#' @section Parameters: See [`LearnerClassifTorch`]
#' Only those from [`LearnerClassifTorch`]
#' @section Fields: `r roxy_fields_inherit(LearnerClassifTorchModel)`
#' @section Methods: `r roxy_methods_inherit(LearnerClassifTorchModel)`
#'
#' @export
LearnerClassifTorchModel = R6Class("LearnerClassifTorchModel",
  inherit = LearnerClassifTorch,
  public = list(
    initialize = function(network, ingress_tokens, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      callbacks = list(), packages = character(0), feature_types = NULL) {
      private$.network_stored = assert_class(network, "nn_module")
      private$.ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken")
      if (is.null(feature_types)) {
        feature_types = mlr_reflections$task_feature_types
      } else {
        assert_subset(feature_types, mlr_reflections$task_feature_types)
      }
      super$initialize(
        id = "classif.torch_model",
        label = "Torch Classification Model",
        optimizer = optimizer,
        loss = loss,
        packages = packages,
        param_set = ps(),
        feature_types = feature_types,
        man = "mlr3torch::mlr_learners_classif.torch_model"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      private$.network_stored
    },
    .dataset = function(task, param_vals) {
      dataset = task_dataset(
        task,
        feature_ingress_tokens = private$.ingress_tokens,
        # TODO: Maybe users should be able to specify the target batchgetter?
        target_batchgetter = target_batchgetter("classif"),
        device = param_vals$device %??% self$param_set$default$device
      )
    },
    .network_stored = NULL,
    .ingress_tokens = NULL
  )
)
