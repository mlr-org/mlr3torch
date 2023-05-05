#' @title Abstract Base Class for a Torch Classification Learner
#'
#' @usage NULL
#' @name mlr_learners_classif.torch
#' @format `r roxy_format(LearnerClassifTorch)`
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch classifiction learners should inherit from the respective subclass.
#'
#' @section Construction:
#' Classification:
#'
#' `r roxy_construction(LearnerClassifTorch)`
#'
#' Regression:
#'
#' `r roxy_construction(LearnerRegrTorch)`
#'
#' * `r roxy_param_id()`
#' * `optimizer` :: [`TorchOptimizer`]\cr
#'   The optimizer for the model.
#' * `loss` :: [`TorchLoss`]\cr
#'   The loss for the model.
#' * `callbacks`:: `list()` of [`TorchCallback`] objects\cr
#'   The callbacks used for training. Must have unique IDs.
#' * `r roxy_param_param_set()`
#' * `properties` :: `character()`\cr
#'   The properties for the learner, see `mlr_reflections$learner_properties`.
#' * `packages` :: `character()`\cr
#'   The additional packages on which the learner depends. The packages `"torch"` and `"mlr3torch"` are automatically
#'   added so do not have to be passed explicitly.
#' * `predict_types` :: `character()`\cr
#'   The learner's predict types, see `mlr_reflections$learner_predict_types`.
#'   The default is `"response"` and `"prob"`.
#' * `feature_types` :: `character()`\cr
#'   The feature types the learner can deal with, see `mlr_reflections$task_feature_types`.
#' * `man` :: `character(1)`\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#' * `label` :: `character(1)`\cr
#'   The label for the learner.
#'
#' @section State:
#' The state is a list with elements `network`, `optimizer`, `loss_fn` and `callbacks`.
#'
#' @template paramset_torchlearner
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload two, methods, namely the
#' `private$.network(task, param_vals)` and the `private$.dataset(task, param_vals)`.
#' The former should construct [`torch::nn_module`] object for the given task and parameter values, while the latter
#' is responsible for creating a [`torch::dataset`].
#' Note that the output of this network are expected to be the scores before the application of the final softmax
#' layer.
#'
#' It is also possible to overwrite the private `$.dataloader()` method, which otherwise calls `$.dataset()` and
#' creates a dataloader from that dataset. When doing so, it is important to respect the parameter `shuffle`, because
#' this method is used to ceate the dataloader for prediction as well.
#'
#' While it is possible to add parameters by specifying the `param_set` construction argument, it is currently
#' not possible to change these parameters.
#' Note that none of the parameters provided in `param_set` can have an id that starts with `"loss."`, `"opt.",
#' or `"cb."`, as these are preserved for the dynamically constructed parameters of the optimizer and the loss
#' function.
#'
#' @section Fields:
#' Fields inherited from [`LearnerClassif`] or [`LearnerRegr`] and
#'
#' * `network` :: ([`nn_module()`][torch::nn_module])\cr
#'   The network (only available after training).
#' * `history` :: [`CallbackTorchHistory`]\cr
#'   The training history.
#' @section Methods: `r roxy_methods_inherit(LearnerClassifTorch)`
#' @section Internals:
#' A [`ParamSetCollection`] is created that combines the `param_set` from the construction with the
#' default torch parameters, as well as the loss, optimizer and callback parameters
#' (prefixed with `"loss."`, `"opt."`, and `"cb."` respectively.
#'
#' @family Learners
#' @export
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassif,
  public = list(
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
    network = function(rhs) learner_torch_network(self, rhs),
    param_set = function(rhs) learner_torch_param_set(self, rhs),
    history = function(rhs) learner_torch_history(self, rhs)
  )
)


#' @title Abstract Base Class for a Torch Learner
#'
#' @usage NULL
#' @name mlr_learners_regr.torch
#' @format `r roxy_format(LearnerRegrTorch)`
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch regression learners should inherit from the respective subclass.
#'
#' @inheritSection mlr_learners_classif.torch Construction
#' @inheritSection mlr_learners_classif.torch State
#' @inheritSection mlr_learners_classif.torch Parameters
#' @inheritSection mlr_learners_classif.torch Fields
#' @section Methods: `r roxy_methods_inherit(LearnerRegrTorch)`
#' @inheritSection mlr_learners_classif.torch Internals
#' @inheritSection mlr_learners_classif.torch Inheriting
#'
#' @family Learners
#' @export
LearnerRegrTorch = R6Class("LearnerRegrTorch",
  inherit = LearnerRegr,
  public = list(
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
    network = function(rhs) learner_torch_network(self, rhs),
    param_set = function(rhs) learner_torch_param_set(self, rhs),
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
