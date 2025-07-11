#' @title Learner Torch Model
#'
#' @name mlr_learners_torch_model
#'
#' @description
#' Create a torch learner from an instantiated [`nn_module()`][torch::nn_module].
#' For classification, the output of the network must be the scores (before the softmax).
#'
#' @template param_task_type
#' @param network ([`nn_module`][torch::nn_module])\cr
#'   An instantiated [`nn_module`][torch::nn_module]. Is not cloned during construction.
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
#' @examplesIf torch::torch_is_installed()
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
#'   epochs = 1,
#'   device = "cpu"
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
    initialize = function(network = NULL, ingress_tokens = NULL, task_type, properties = NULL, optimizer = NULL, loss = NULL,
      callbacks = list(), packages = character(0), feature_types = NULL) {
      # we need to serialize here as otherwise encapsulation and parallelization fails
      if (!is.null(network)) private$.network_stored = torch_serialize(assert_class(network, "nn_module"))
      if (!is.null(ingress_tokens)) self$ingress_tokens = ingress_tokens
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
        loss = loss,
        optimizer = optimizer,
        callbacks = callbacks,
        task_type = task_type,
        label = "Torch Model",
        properties = properties,
        packages = packages,
        param_set = ps(),
        feature_types = feature_types,
        jittable = TRUE,
        man = "mlr3torch::mlr_learners.torch_model"
      )
    }
  ),
  active = list(
    #' @field ingress_tokens (named `list()` with `TorchIngressToken` or `NULL`)\cr
    #' The ingress tokens. Must be non-`NULL` when calling `$train()`.
    ingress_tokens = function(rhs) {
      if (!missing(rhs)) {
        private$.ingress_tokens_ = assert_list(rhs, types = "TorchIngressToken", min.len = 1L, names = "unique")
      }
      private$.ingress_tokens_
    }
  ),
  private = list(
    .ingress_tokens_ = NULL,
    deep_clone = function(name, value) {
      if (name == ".network_stored" && is.null(value) && !is.null(self$state)) {
        # the initial network state is lost after training a LearnerTorchModel
        stopf("Learner %s: Can only create deep clone for untrained learner", self$id)
      } else {
        super$deep_clone(name, value)
      }
    },
    .network = function(task, param_vals) {
      if (is.null(private$.network_stored)) {
        stopf("No network stored, did you already train learner '%s' or did not specify a model?", self$id)
      }
      network = if (test_class(private$.network_stored, "nn_module")) {
        # optimization for PipeOpTorchModel, where we control the construction of LearnerTorchModel
        private$.network_stored
      } else {
        torch_load(private$.network_stored)
      }
      private$.network_stored = NULL
      network
    },
    .dataset = function(task, param_vals) {
      ingress_tokens = self$ingress_tokens
      if (is.null(ingress_tokens)) {
        stopf("Learner '%s' has no $ingress_tokens set.", self$id)
      }
      dataset = task_dataset(
        task,
        feature_ingress_tokens = ingress_tokens,
        target_batchgetter = get_target_batchgetter(task)
      )
    },
    .network_stored = NULL,
    .additional_phash_input = function() {
      list(self$properties, self$feature_types, private$.network_stored, self$packages, private$.ingress_tokens)
     }
  )
)

#' @include PipeOpTorchIngress.R task_dataset.R
register_learner("classif.torch_model", LearnerTorchModel)
register_learner("regr.torch_model", LearnerTorchModel)
