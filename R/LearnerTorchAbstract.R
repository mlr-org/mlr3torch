#' @title Abstract Base Class for a Torch Network
#'
#' @usage NULL
#' @name mlr_learners.torch_abstract
#'
#' @format [`R6Class`] object inheriting from [`LearnerClassif`] or [`LearnerRegr`] / [`Learner`].
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All Torch earners should inherit from the respective subclass.
#'
#' @section Construction:
#' `r roxy_construction(LearnerClassifTorchAbstract)`
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
#'   The additional packages on which the learner depends.
#' * `predict_types` :: (`character()`)\cr
#'   The learner's predict types, see `mlr_reflections$learner_predict_types`.
#' * `feature_types` :: (`character()`)\cr
#'   The feature types the learner can deal with, see `mlr_reflections$task_feature_types`.
#' * `man` :: (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#' * `label` :: (`character(1)`)\cr
#'   The label for the learner.
#'
#' @template paramset_torchlearner
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload the `private$.network(task)` method which should construct
#' a `nn_module()` object for the given task and parameter values. Note that the parameters must not start with
#' `"loss."` or `"opt."`, as these prefixes are reserverd for the dynamically constructed parameters of the optimizer
#' and the loss function.
#'
#' @section Fields:
#' Fields inherited from [`LearnerClassif`] or [`LearnerRegr`] and
#'
#' * `network` :: ([`nn_module()`][torch::nn_module])\cr
#'   The network (only available after training).
#' @section Methods: Only methods inherited from [`LearnerRegr`] / [`Learner`].
#' @section Internals:
#' A [`ParamSetCollection`] is created that combines the `param_set` from the construction with the
#' default parameters obtained by [`paramset_torchlearner()`], as well as the loss and optimizer parameter
#' (prefixed with `"loss."` and `"opt."` respectively. Therefore
#' TODO: write more here.
#'
#' @family Learners
#' @export
LearnerClassifTorchAbstract = R6Class("LearnerClassifTorchAbstract",
  inherit = LearnerClassif,
  public = list(
    initialize = function(id, optimizer, loss, param_set, properties = NULL, packages = character(0),
      predict_types, feature_types, man, label) {
      private$.optimizer = as_torch_optimizer(optimizer, clone = TRUE)
      private$.optimizer$param_set$set_id = "opt"

      private$.loss = as_torch_loss(loss, clone = TRUE)
      private$.loss$param_set$set_id = "loss"

      assert_subset("classif", private$.loss$task_types)
      assert_subset(properties, mlr_reflections$learner_properties[["classif"]])
      assert_subset(predict_types, names(mlr_reflections$learner_predict_types[["classif"]]))
      assert_true(!any(grepl("^(loss\\.|optimizer\\.)", param_set$ids())))

      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3torch", "torch"), packages)

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
    .network = function(task) stop(".network must be implemented."),
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
    .dataloader = function(task, param_vals) stop(".dataloader must be implemented."),
    .optimizer = NULL,
    .loss = NULL,
    .param_set_base = NULL,
    deep_clone = function(name, value) deep_clone(self, name, value)
  ),
  active = list(
    network = function(rhs) learner_torch_network(self, rhs),
    param_set = function(rhs) learner_torch_param_set(self, rhs)
  )
)


deep_clone = function(self, name, value) {
  super = self$.__enclos_env__$super

  if (name == "state") {
    state = super$deep_clone("state", value)
    model = state$model
    if (!is.null(model)) {
      state$model = insert_named(state$model, list(
        network = model$network$clone(deep = TRUE),
        optimizer = model$optimizer$clone(deep = TRUE),
        loss = model$loss$clone(deep = TRUE),
        callbacks = map(model$callbacks, function(x) x$clone(deep = TRUE))
      ))
    }
    return(state)
  } else if (name == "param_set") {
    # Otherwise references get lost, i.e. the paramset of the TorchLoss and TorchOptimizer
    # are no longer identical to the parameter sets contained in the collection
    return(NULL)
  }
}
