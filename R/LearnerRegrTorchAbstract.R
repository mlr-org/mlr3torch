#' @title Abstract Base Class for Torch Regression Network
#' @description
#' All Torch Regression Learners should inherit from this base class.
#' It implements basic functionality that can be reused for all sort of learners
#' It is not intended for direct use.
#'
#' @param id (`character(1)`)\cr
#'   The id for the learner.
#' @param optimizer (`character(1)`)\cr
#'   The optimizer, see `torch_reflections$optimizer`.
#' @param loss (`character(1)`)\cr
#'   The loss, see `torch_reflections$loss$regr`.
#' @param param_set (`paradox::ParamSet`)\cr
#'   Additional parameters to the standard paramset created by `make_paramset()`.
#' @param label (`character(1)`)\cr
#'   The label for the learner.
#' @param properties (`character()`)\cr
#'   The properties for the learner, see `mlr_reflections$learner_properties`.
#' @param packages (`character()`)\cr
#'   The additional packages on which the learner depends.
#' @param predict_types (`character()`)\cr
#'   The learner's predict types, see `mlr_reflections$learner_predict_types`.
#' @param feature_types (`character()`)\cr
#'   The feature types the learner can deal with, see `mlr_reflections$task_feature_types`.
#' @param man (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#'
#' @export
LearnerRegrTorchAbstract = R6Class("LearnerRegrTorchAbstract",
  inherit = LearnerRegr,
  public = list(
    #' @description Initializes an object of this [R6][R6::R6Class] class.
    initialize = function(id, optimizer, loss, param_set = ps(), label = NULL, properties = NULL,
      packages = character(0), predict_types = NULL, feature_types, man) {
      private$.optimizer = assert_choice(optimizer, torch_reflections$optimizer)
      private$.loss = assert_choice(loss, torch_reflections$loss$regr)
      # FIXME: loglik?
      properties = properties %??% c("weights", "multiclass", "twoclass", "hotstart_forward")
      predict_types =  predict_types %??% "response"
     label = label %??% "Neural Network Regression Model"

      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3torch", "torch"), packages)
      # note that we don't have to explicitly check that the optimizer params are disjunct from
      # the remaining parameters as this is done here anyway (call fails if it doesn't).
      param_set_complete = make_paramset("regr", optimizer, loss, architecture = TRUE)
      param_set_complete$add(param_set)

      super$initialize(
        id = id,
        packages = packages,
        param_set = param_set_complete,
        predict_types = predict_types,
        properties = properties,
        data_formats = "data.table",
        label = label,
        feature_types = feature_types,
        man = man
      )
    },
    #' @description Builds the model `list(network, optimizer, loss_fn, history)`.
    #' @param task ([`Task`][mlr3::Task])\cr
    #'   The task for which to build the network.
    build = function(task) {
      network = private$.network(task)
      model = build_torch(self, task, network)
      return(model)
    }
  ),
  private = list(
    .train = function(task) {
      model = self$build(task)
      learner_torch_train(self, model, task)
    },
    .predict = function(task) {
      # When keep_last_prediction = TRUE we store the predictions of the last validation and we
      # therefore don't have to recompute them in the resample(), but can simple return the
      # cached predictions
      learner_torch_predict(self, task)
    },
    .optimizer = NULL,
    .loss = NULL
  ),
  active = list(
    #' @field parameters (`list()`)\cr
    #'   A list with the network's parameters.
    parameters = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network$parameters
    },
    #' @field history ([`History][History])\cr
    #'   History of the training proceess.
    history = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$history
    },
    #' @field optimizer ([`torch_Optimizer`][torch::optimizer])\cr
    #'  The optimizer.
    optimizer = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$optimizer
    },
    #' @field loss_fn (`nn_loss()`)\cr
    #'   The loss function.
    loss_fn = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$loss_fn
    },
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #'   The network.
    network = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network
    }
  )
)
