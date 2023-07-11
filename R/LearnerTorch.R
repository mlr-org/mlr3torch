#' @title Base Class for Torch Learners
#'
#' @name mlr_learners_torch
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch learners should inherit from this class.
#'
#' It also allows to hook into the training loop via a callback mechanism.
#'
#' @template param_id
#' @template param_task_type
#' @template param_param_vals
#' @template param_param_set
#' @template param_properties
#' @template param_packages
#' @template param_feature_types
#' @template param_man
#' @template param_label
#' @param predict_types (`character()`)\cr
#'   The predict types.
#'   See [`mlr_reflections$learner_predict_types`][mlr_reflections] for available values.
#'   For regression, the default is `"response"`.
#'   For classification, this defaults to `"response"` and `"prob"`.
#'   To deviate from the defaults, it is necessary to overwrite the private `$.encode_prediction()`
#'   method, see section *Inheriting*.
#' @param loss (`NULL` or [`TorchLoss`])\cr
#'   The loss to use for training.
#'   Defaults to MSE for regression and cross entropy for classification.
#' @param optimizer (`NULL` or [`TorchOptimizer`])\cr
#'   The optimizer to use for training.
#'   Defaults to adam.
#' @param callbacks (`list()` of [`TorchCallback`]s)\cr
#'   The callbacks to use for training.
#'   Defaults to an empty` list()`, i.e. no callbacks.
#'
#' @section State:
#' The state is a list with elements `network`, `optimizer`, `loss_fn`, `callbacks` and `seed`.
#'
#' @template paramset_torchlearner
#'
#' @section Inheriting:
#' There are no seperate classes for classification and regression to inherit from.
#' Instead, the `task_type` must be specified  as a construction argument.
#' Currently, only classification and regression are supported.
#'
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
#'   Create the dataset for the task.
#'   Must respect the parameter value of the device.
#'   Moreover, one needs to pay attention respect the row ids of the provided task.
#'
#' It is also possible to overwrite the private `.dataloader()` method instead of the `.dataset()` method.
#' Per default, a dataloader is constructed using the output from the `.dataset()` method.
#'
#' * `.dataloader(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`torch::dataloader`]\cr
#'   Create a dataloader from the task.
#'   Needs to respect at least `batch_size` and `shuffle` (otherwise predictions can be permuted).
#'
#' To change the predict types, the private `.encode_prediction()` method can be overwritten:
#'
#' * `.encode_prediction(predict_tensor, task, param_vals)`\cr
#'   ([`torch_tensor`], [`Task`], `list()`) -> `list()`\cr
#'   Take in the raw predictions from `self$network` (`predict_tensor`) and encode them into a
#'   format that can be converted to valid `mlr3` predictions using [`mlr3::as_prediction_data()`].
#'   This method must take `self$predict_type` into account.
#'
#' While it is possible to add parameters by specifying the `param_set` construction argument, it is currently
#' not possible to remove existing parameters, i.e. those listed in section *Parameters*.
#' None of the parameters provided in `param_set` can have an id that starts with `"loss."`, `"opt.",
#' or `"cb."`, as these are preserved for the dynamically constructed parameters of the optimizer, the loss function,
#' and the callbacks.
#'
#' @family Learner
#' @export
LearnerTorch = R6Class("LearnerTorch",
  inherit = Learner,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, task_type, param_set, properties, man, label, feature_types,
      optimizer = NULL, loss = NULL, packages = NULL, predict_types = NULL, callbacks = list()) {
      assert_choice(task_type, c("regr", "classif"))

      learner_torch_initialize(self = self, private = private, super = super,
        task_type = task_type,
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
  active = list(
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #'   The network (only available after training).
    network = function(rhs) {
      assert_ro_binding(rhs)
      if (is.null(self$state)) {
        stopf("Cannot access network before training.")
      }
      self$state$model$network
    },
    #' @field param_set ([`ParamSet`])\cr
    #'   The parameter set
    param_set = function(rhs) {
      if (is.null(private$.param_set)) {
        private$.param_set = ParamSetCollection$new(c(
          list(private$.param_set_base, private$.optimizer$param_set, private$.loss$param_set),
          map(private$.callbacks, "param_set"))
        )
      }
      private$.param_set
    },
    #' @field history ([`CallbackSetHistory`])\cr
    #' Shortcut for `learner$model$callbacks$history`.
    history = function(rhs) {
      assert_ro_binding(rhs)
      if (is.null(self$state)) {
        stopf("Cannot access history before training.")
      }
      if (is.null(self$model$callbacks$history)) {
        stopf("No history found. Did you specify t_clbk(\"history\") during construction?")
      }
      self$model$callbacks$history
    }
  ),
  private = list(
    .train = function(task) {
      param_vals = self$param_set$get_values(tags = "train")
      param_vals$device = auto_device(param_vals$device)
      if (param_vals$seed == "random") param_vals$seed = sample.int(10000000L, 1L)

      with_torch_settings(seed = param_vals$seed, num_threads = param_vals$num_threads, {
        learner_torch_train_worker(self, private, super, task, param_vals)
      })
    },
    .predict = function(task) {
      param_vals = self$param_set$get_values(tags = "predict")
      param_vals$device = auto_device(param_vals$device)

      with_torch_settings(seed = self$model$seed, num_threads = param_vals$num_threads, {
        self$network$eval()
        data_loader = private$.dataloader_predict(task, param_vals)
        predict_tensor = torch_network_predict(self$network, data_loader)
        private$.encode_prediction(predict_tensor, task, param_vals)
      })
    },
    .encode_prediction = function(predict_tensor, task, param_vals) {
      encode_prediction(predict_tensor, self$predict_type, task)
    },
    .network = function(task, param_vals) stop(".network must be implemented."),
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
   .dataloader = function(task, param_vals) {
      learner_torch_dataloader(self, task, param_vals)
    },
    .dataloader_predict = function(task, param_vals) {
      learner_torch_dataloader_predict(self, task, param_vals)
    },
    .dataset = function(task, param_vals) stop(".dataset must be implemented."),
    .optimizer = NULL,
    .loss = NULL,
    .param_set_base = NULL,
    .callbacks = NULL,
    deep_clone = function(name, value) deep_clone(self, private, super, name, value)
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
