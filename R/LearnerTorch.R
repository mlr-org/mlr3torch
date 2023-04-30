#' @title Abstract Base Class for a Torch Network
#'
#' @usage NULL
#' @name mlr_learners_classif.torch
#'
#' @format `r roxy_format(LearnerClassifTorch)`
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch learners should inherit from the respective subclass.
#' To create a torch learner from a [`nn_module`], use [`LearnerClassifTorch`] or [`LearnerRegrTorch`] instead.
#'
#' @section Construction:
#' `r roxy_construction(LearnerClassifTorch)`
#'
#' * `r roxy_param_id()`
#' * `optimizer` :: ([`TorchOptimizer`])\cr
#'   The optimizer for the model.
#' * `loss` :: ([`TorchLoss`])\cr
#'   The loss for the model.
#' * `r roxy_param_param_set()`
#' * `properties` :: (`character()`)\cr
#'   The properties for the learner, see `mlr_reflections$learner_properties`.
#' * `packages` :: (`character()`)\cr
#'   The additional packages on which the learner depends. The packages `"torch"` and `"mlr3torch"` are automatically
#'   added so do not have to be passed explicitly.
#' * `predict_types` :: (`character()`)\cr
#'   The learner's predict types, see `mlr_reflections$learner_predict_types`.
#'   The default is `"response"` and `"prob"`.
#' * `feature_types` :: (`character()`)\cr
#'   The feature types the learner can deal with, see `mlr_reflections$task_feature_types`.
#' * `man` :: (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#' * `label` :: (`character(1)`)\cr
#'   The label for the learner.
#'
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
    initialize = function(id, optimizer, loss, param_set, properties = NULL, packages = character(0),
      predict_types = c("response", "prob"), feature_types, man, label, callbacks = list()) {
      private$.optimizer = as_torch_optimizer(optimizer, clone = TRUE)
      private$.optimizer$param_set$set_id = "opt"

      private$.loss = as_torch_loss(loss, clone = TRUE)
      private$.loss$param_set$set_id = "loss"

      callbacks = as_torch_callbacks(callbacks, clone = TRUE)
      callback_ids = ids(callbacks)
      assert_names(callback_ids, type = "unique")
      if ("history" %in% callback_ids) {
        stopf("Callback with id 'history' is reserved for CallbackTorchHistory, which is always added.")
      }

      callbacks = c(t_clbk("history"), callbacks)
      private$.callbacks = set_names(callbacks, ids(callbacks))
      walk(private$.callbacks, function(cb) {
        cb$param_set$set_id = paste0("cb.", cb$id)
      })

      # TODO: Here we should tag all the parameters of the callbacks and optimizer and loss with `"train"` (?)

      packages = unique(c(
        packages,
        unlist(map(private$.callbacks, "packages")),
        private$.loss$packages,
        private$.optimizer$packages
      ))

      properties = properties %??% c("twoclass", "multiclass")

      assert_subset(properties, mlr_reflections$learner_properties[["classif"]])
      assert_subset(predict_types, names(mlr_reflections$learner_predict_types[["classif"]]))
      assert_true(!any(grepl("^(loss\\.|opt\\.|cb\\.)", param_set$ids())))
      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3", "mlr3torch", "torch"), packages)

      paramset_torch = paramset_torchlearner()
      if (param_set$length > 0) {
        private$.param_set_base = ParamSetCollection$new(list(param_set, paramset_torch))
      } else {
        private$.param_set_base = paramset_torch
      }

      super$initialize(
        id = id,
        packages = packages,
        param_set = self$param_set,
        predict_types = predict_types,
        properties = properties,
        data_formats = "data.table",
        label = label,
        feature_types = feature_types,
        man = man
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
