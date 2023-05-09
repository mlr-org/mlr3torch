#' @title Classification Torch Learner
#'
#' @name mlr_learners_classif.torch_model
#'
#' @description
#' Create a classification learner from an instantiated [`nn_module()`].
#' This is learner is used internally by [`PipeOpTorchModelClassif`].
#'
#' The output of the network must be the scores (before the softmax).
#'
#' @param network ([`nn_module`])\cr
#'   An instantiated [`nn_module`]. Is not cloned during construction.
#'   Outputs must be the scores (before the softmax).
#' @param ingress_tokens (`list` of [`TorchIngressToken()`])\cr
#'   A list with ingress tokens that defines how the dataloader will be defined.
#' @template param_optimizer
#' @template param_loss
#' @template param_callbacks
#' @template param_packages
#' @param feature_types (`character()`)\cr
#'   The feature types. Defaults to all available feature types.
#'
#' @section Parameters: See [`LearnerClassifTorch`]
#' @family Learner
#' @family Graph Network
#' @include LearnerTorch.R
#' @export
#' @examples
#'
#' # The iris task has 4 features and 3 classes
#' network = nn_linear(4, 3)
#' task = tsk("iris")
#'
#' # This defines the dataloader.
#' # It loads all 4 features, which are also numeric.
#' # The shape is (NA, 4) because the batch dimension is generally NA
#' ingress_tokens = list(
#'   input = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4))
#' )
#'
#' # Creating the learner and setting required parameters
#' learner = LearnerClassifTorchModel$new(network, ingress_tokens)
#' learner$param_set$set_values(batch_size = 16, epochs = 1)
#'
#' # A simple train-predict
#' ids = partition(task)
#' learner$train(task, ids$train)
#' learner$predict(task, ids$test)
LearnerClassifTorchModel = R6Class("LearnerClassifTorchModel",
  inherit = LearnerClassifTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(network, ingress_tokens, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      callbacks = list(), packages = character(0), feature_types = NULL) {
      # TODO: What about the learner properties?
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
#' @name mlr_learners_regr.torch_model
#'
#' @description
#' Create a regression learner from an instantiated [`nn_module()`].
#' This is learner is used internally by [`PipeOpTorchModelRegr`].
#'
#' @param network ([`nn_module`])\cr
#'   An instantiated [`nn_module`]. This is not cloned during construction.
#'   Outputs must be the scores (before the softmax).
#' @param ingress_tokens (`list` of [`TorchIngressToken()`])\cr
#'   A list with ingress tokens that defines how the dataloader will be defined.
#' @template param_optimizer
#' @template param_loss
#' @template param_callbacks
#' @template param_packages
#' @param feature_types (`character()`)\cr
#'   The feature types. Defaults to all available feature types.
#'
#' @section Parameters: See [`LearnerRegrTorch`]
#' @family Learner
#' @family Graph Network
#' @include LearnerTorch.R
#' @export
#' @examples
#'
#' # The mtcars task has 10 features
#' # The output of the network is 1, as it is a regression problem
#' network = nn_linear(10, 1)
#' task = tsk("mtcars")
#'
#' # This defines the dataloader.
#' # It loads all 10 features, which are also numeric.
#' # The shape is (NA, 10) because the batch dimension is generally NA
#' ingress_tokens = list(
#'   input = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 10))
#' )
#'
#' # Creating the learner and setting required parameters
#' learner = LearnerRegrTorchModel$new(network, ingress_tokens)
#' learner$param_set$set_values(batch_size = 16, epochs = 1)
#'
#' # A simple train-predict
#' ids = partition(task)
#' learner$train(task, ids$train)
#' learner$predict(task, ids$test)
LearnerRegrTorchModel = R6Class("LearnerRegrTorchModel",
  inherit = LearnerRegrTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
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
        label = "Torch Regression Model",
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
