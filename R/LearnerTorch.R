#' @title Abstract Base Class for a Torch Classification Learner
#'
#' @name mlr_learners_classif.torch
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch classifiction learners should inherit from the respective class, i.e.
#' [`LearnerClassifTorch`] for classification and [`LearnerRegrTorch`] for regression.
#'
#' It also allows to hook into the training loop via a callback mechanism.
#'
#' @section State:
#' The state is a list with elements `network`, `optimizer`, `loss_fn` and `callbacks`.
#'
#' @template paramset_torchlearner
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload two private methods:
#'
#' * `.network(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`nn_module`]\cr
#'   Construct a [`torch::nn_module`] object for the given task and parameter values, i.e. the neural network that
#'   is trained by the learner.
#'   For classification, the output of this network are expected to be the scores before the application of the
#'   final softmax layer.
#' * `.dataset(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`torch::dataset`]\cr
#'   Create the dataset for the task.  Must at least respect parameters `batch_size` and `shuffle`.
#'
#' It is also possible to overwrite the private `.dataloader()` method instead of the `.dataset()` method.
#' Per default, a dataloader is constructed using the output from the `.dataset()` method.
#'
#' * `.dataloader(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`torch::dataloader`]\cr
#'   Create a dataloader from the task.
#'   Needs to respect at least `batch_size` and `shuffle` (otherwise predictions are permuted).
#'
#' While it is possible to add parameters by specifying the `param_set` construction argument, it is currently
#' not possible to remove existing parameters, i.e. those listed in section *Parameters*.
#' None of the parameters provided in `param_set` can have an id that starts with `"loss."`, `"opt.",
#' or `"cb."`, as these are preserved for the dynamically constructed parameters of the optimizer, the loss function,
#' and the callbacks.
#'
#' @family Learner
#' @export
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassif,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @template param_param_vals
    #' @template param_optimizer
    #' @template param_loss
    #' @template param_param_set
    #' @template param_properties
    #' @template param_packages
    #' @template param_predict_types
    #' @template param_feature_types
    #' @template param_man
    #' @template param_label
    #' @template param_callbacks
    initialize = function(id, optimizer, loss, param_set, properties = c("twoclass", "multiclass"), packages = character(0),
      predict_types = c("response", "prob"), feature_types, man, label, callbacks = list()) {

      learner_torch_initialize(self = self, private = private, super = super,
        task_type = "classif",
        id = id,
        optimizer = optimizer,
        loss = loss,
        param_set = param_set,
        properties = properties,
        packages = packages,
        predict_types = predict_types,
        feature_types = feature_types,
        man = man,
        label = label,
        callbacks = callbacks
      )
    }
  ),
  private = list(
    .train = function(task) {
      learner_torch_train(self, task)
    },
    .predict = function(task) {
      learner_torch_predict(self, task)
    },
    .network = function(task, param_vals) stop(".network must be implemented."),
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
    .dataloader = function(task, param_vals) {
      dataloader(
        private$.dataset(task, param_vals),
        batch_size = param_vals$batch_size %??% self$param_set$default$batch_size,
        shuffle = param_vals$shuffle %??% self$param_set$default$shuffle
      )
    },
    .dataset = function(task, param_vals) stop(".dataset must be implemented."),
    .optimizer = NULL,
    .loss = NULL,
    .param_set_base = NULL,
    .callbacks = NULL,
    deep_clone = function(name, value) deep_clone(self, private, super, name, value)
  ),
  active = list(
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #'   The network (only available after training).
    network = function(rhs) learner_torch_network(self, rhs),
    #' @field param_set ([`ParamSet`])\cr
    #'   The parameter set
    param_set = function(rhs) learner_torch_param_set(self, rhs),
    #' @field history ([`CallbackSetHistory`])\cr
    #' Shortcut for `learner$model$callbacks$history`.
    history = function(rhs) learner_torch_history(self, rhs)
  )
)


#' @title Abstract Base Class for a Torch Regression Learner
#'
#' @name mlr_learners_regr.torch
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch regression learners should inherit from the respective subclass.
#'
#' @inheritSection mlr_learners_classif.torch State
#' @inheritSection mlr_learners_classif.torch Parameters
#' @inheritSection mlr_learners_classif.torch Inheriting
#'
#' @family Learner
#' @export
LearnerRegrTorch = R6Class("LearnerRegrTorch",
  inherit = LearnerRegr,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @template param_param_vals
    #' @template param_optimizer
    #' @template param_loss
    #' @template param_param_set
    #' @template param_properties
    #' @template param_packages
    #' @template param_predict_types
    #' @template param_feature_types
    #' @template param_man
    #' @template param_label
    #' @template param_callbacks
    initialize = function(id, optimizer, loss, param_set, properties = character(0), packages = character(0),
      predict_types = "response", feature_types, man, label, callbacks = list()) {
      learner_torch_initialize(self = self, private = private, super = super,
        task_type = "regr",
        id = id,
        optimizer = optimizer,
        loss = loss,
        param_set = param_set,
        properties = properties,
        packages = packages,
        predict_types = predict_types,
        feature_types = feature_types,
        man = man,
        label = label,
        callbacks = callbacks
      )
    }
  ),
  private = list(
    .train = function(task) {
      learner_torch_train(self, task)
    },
    .predict = function(task) {
      learner_torch_predict(self, task)
    },
    .network = function(task, param_vals) stop(".network must be implemented."),
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
    .dataloader = function(task, param_vals) {
      dataloader(
        private$.dataset(task, param_vals),
        batch_size = param_vals$batch_size %??% self$param_set$default$batch_size,
        shuffle = param_vals$shuffle %??% self$param_set$default$shuffle
      )
    },
    .dataset = function(task, param_vals) stop(".dataset must be implemented."),
    .optimizer = NULL,
    .loss = NULL,
    .param_set_base = NULL,
    .callbacks = NULL,
    deep_clone = function(name, value) deep_clone(self, private, super, name, value)
  ),
  active = list(
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #'   The network (only available after training).
    network = function(rhs) learner_torch_network(self, rhs),
    #' @field param_set ([`ParamSet`])\cr
    #'   The parameter set
    param_set = function(rhs) learner_torch_param_set(self, rhs),
    #' @field history ([`CallbackSetHistory`])\cr
    #'   Shortcut for `learner$model$callbacks$history`.
    history = function(rhs) learner_torch_history(self, rhs)
  )
)


deep_clone = function(self, private, super, name, value) {
  private$.param_set = NULL # required to keep clone identical to original, otherwise tests get really ugly

  if (name == "state") {
    # https://github.com/mlr-org/mlr3torch/issues/97
    if (!is.null(value)) {
      stopf("Deep clone of trained network is currently not supported.")
    } else {
      # Note that private methods are available in super.
      super$deep_clone(name, value)
    }
  } else if (name == ".param_set") {
    # Otherwise the value$clone() is called on NULL which errs
    NULL
  } else {
    # Note that private methods are available in super.
    super$deep_clone(name, value)
  }
}
