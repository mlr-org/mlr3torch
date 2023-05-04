#' @title Classification Torch Learner
#'
#' @usage NULL
#' @name mlr_learners_classif.torch_model
#' @format `r roxy_format(LearnerClassifTorchModel)`
#'
#' @description
#' Classification Torch Learner that is used internally by [`PipeOpTorchModelClassif`].
#'
#' @section Construction:
#'
#' Classification:
#'
#' `r roxy_construction(LearnerClassifTorchModel)`
#'
#' Regression:
#'
#' `r roxy_construction(LearnerRegrTorchModel)`
#'
#' * `network` :: [`nn_module`]\cr
#'   An instantiatded [`nn_module`].
#' * `ingress_tokens` :: `list` of [`TorchIngressToken()`]\cr
#'   A list with ingress tokens that defines how the dataloader will be defined.
#' * `optimizer` :: ([`TorchOptimizer`])\cr
#'   The optimizer for the model. Defaults is adam.
#' * `loss` :: ([`TorchLoss`])\cr
#'   The loss for the model. Default is cross entropy.
#' * `callbacks`:: `list()` of [`TorchCallback`] objects\cr
#'   The callbacks used for training. Must have unique IDs.
#' * `packages` :: (`character()`)\cr
#'   The additional packages on which the learner depends.
#'   added so do not have to be passed explicitly.
#' * `feature_types` :: (`character()`)\cr
#'   The feature types the learner can deal with, see `mlr_reflections$task_feature_types`.
#'
#'
#' @section State: See [`LearnerClassif`]
#' @section Parameters: See [`LearnerClassifTorch`]
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
        target_batchgetter = target_batchgetter("classif"),
        device = param_vals$device %??% self$param_set$default$device
      )
    },
    .network_stored = NULL,
    .ingress_tokens = NULL
  )
)


#' @title Regression Torch Learner
#'
#' @usage NULL
#' @name mlr_learners_regr.torch_model
#' @format `r roxy_format(LearnerRegrTorchModel)`
#'
#' @description
#' Regression Torch Learner that is used internally by [`PipeOpTorchModelRegr`].
#'
#' @inheritSection mlr_learners_classif.torch_model Construction
#'
#' @section State: See [`LearnerRegrTorch`]
#' @section Parameters: See [`LearnerRegrTorch`]
#' @section Fields: `r roxy_fields_inherit(LearnerRegrTorchModel)`
#' @section Methods: `r roxy_methods_inherit(LearnerRegrTorchModel)`
#'
#' @export
LearnerRegrTorchModel = R6Class("LearnerRegrTorchModel",
  inherit = LearnerRegrTorch,
  public = list(
    initialize = function(network, ingress_tokens, optimizer = t_opt("adam"), loss = t_loss("mse"),
      callbacks = list(), packages = character(0), feature_types = NULL) {
      private$.network_stored = assert_class(network, "nn_module")
      private$.ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken")
      if (is.null(feature_types)) {
        feature_types = mlr_reflections$task_feature_types
      } else {
        assert_subset(feature_types, mlr_reflections$task_feature_types)
      }
      super$initialize(
        id = "regr.torch_model",
        label = "Torch Degression Model",
        optimizer = optimizer,
        loss = loss,
        packages = packages,
        param_set = ps(),
        feature_types = feature_types,
        man = "mlr3torch::mlr_learners_regr.torch_model"
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
        target_batchgetter = target_batchgetter("regr"),
        device = param_vals$device %??% self$param_set$default$device
      )
    },
    .network_stored = NULL,
    .ingress_tokens = NULL
  )
)
