#' @title Learner Torch Model
#'
#' @name mlr_learners_torch_model
#'
#' @description
#' Create a torch learner from an instantiated [`nn_module()`].
#' For classification, the output of the network must be the scores (before the softmax).
#'
#' @template param_task_type
#' @param network ([`nn_module`])\cr
#'   An instantiated [`nn_module`]. Is not cloned during construction.
#'   For classification, outputs must be the scores (before the softmax).
#' @param ingress_tokens (`list` of [`TorchIngressToken()`])\cr
#'   A list with ingress tokens that defines how the dataloader will be defined.
#' @template param_optimizer
#' @template param_loss
#' @template param_callbacks
#' @template param_packages
#' @param feature_types (`NULL` or `character()`)\cr
#'   The feature types. Defaults to all available feature types.
#' @param properties (`NULL` or `character()`)\cr
#'   The properties of the learner.
#'   Defaults to all available properties for the given task type.
#' @section Parameters: See [`LearnerTorch`]
#' @family Learner
#' @family Graph Network
#' @include LearnerTorch.R
#' @export
#' @examples
#' # We show the learner using a classification task
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
#' learner = lrn("classif.torch_model",
#'   network = network,
#'   ingress_tokens = ingress_tokens,
#'   batch_size = 16,
#'   epochs = 1
#' )
#'
#' # A simple train-predict
#' ids = partition(task)
#' learner$train(task, ids$train)
#' learner$predict(task, ids$test)
LearnerTorchModel = R6Class("LearnerTorchModel",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, network, ingress_tokens, properties = NULL, optimizer = NULL, loss = NULL,
      callbacks = list(), packages = character(0), feature_types = NULL) {
      # TODO: What about the learner properties?
      private$.network_stored = assert_class(network, "nn_module")
      private$.ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken")
      if (is.null(feature_types)) {
        feature_types = unname(mlr_reflections$task_feature_types)
      } else {
        assert_subset(feature_types, mlr_reflections$task_feature_types)
      }
      if (is.null(properties)) {
        properties = mlr_reflections$learner_properties[[task_type]]
      } else {
        properties = assert_subset(properties, mlr_reflections$learner_properties[[task_type]])
      }
      super$initialize(
        id = paste0(task_type, ".model"),
        task_type = task_type,
        label = "Torch Model",
        optimizer = optimizer,
        properties = properties,
        loss = loss,
        packages = packages,
        param_set = ps(),
        feature_types = feature_types,
        man = "mlr3torch::mlr_learners.torch_model"
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
        target_batchgetter = get_target_batchgetter(self$task_type),
        device = param_vals$device
      )
    },
    .network_stored = NULL,
    .ingress_tokens = NULL
  )
)

register_learner("classif.torch_model", LearnerTorchModel)
register_learner("regr.torch_model", LearnerTorchModel)
