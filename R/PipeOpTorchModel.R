#' @title Torch Model
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model
#' @format `r roxy_format(PipeOpTorchModel)`
#'
#' @description
#' Builds a Torch Learner from a [`ModelDescriptor`] and trains it with the given parameter specification.
#' For a specific task type, use [`PipeOpTorchModelClassif`] or [`PipeOpTorchModelRegr`].
#'
#' @section Construction: `r roxy_construction(PipeOpTorchModel)`
#' * `r roxy_param_id("torch_model")`
#' * `r roxy_param_param_vals()`
#' * `task_type` :: `character(1)`\cr
#'   The task type of the model. Currently `"regr"` and "`classif`" are supported.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"` that takes in `ModelDescriptor` during traing and a `Task` of the specified
#' `task_type` during prediction.
#' The output is `NULL` during training and a `Prediction` of given `task_type` during prediction.
#'
#' @section State:
#' A trained `LearnerRegrTorchModel` `LearnerClassifTorchModel`.
#'
#' @template paramset_torchlearner
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#'
#' @section Internals:
#' First a [`nn_graph`] is created by calling [`model_descriptor_to_module()`] and than a
#' [`LearnerClassifTorchModel`] or [`LearnerRegrTorchModel`] is created from the `nn_graph` in combination with the
#' information stored in the [`ModelDescriptor`].
#' Then the parameters are set according to the parameters specified in `PipeOpTorchModel` and its '$train()` method
#' is called on the [`Task`] stored in the [`ModelDescriptor`].
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchModel = R6Class("PipeOpTorchModel",
  inherit = PipeOp,
  public = list(
    initialize = function(id = "torch_model", param_vals = list(), task_type) {
      private$.task_type = assert_choice(task_type, c("classif", "regr"))
      param_set = paramset_torchlearner()
      input = data.table(
        name = "input",
        train = "ModelDescriptor",
        predict = mlr_reflections$task_types["classif", task]
      )
      output = data.table(
        name = "output",
        train = "NULL",
        predict = mlr_reflections$task_types["classif", prediction]
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      md = inputs[[1]]

      if (is.null(md$loss)) {
        stopf("No loss configured in ModelDescriptor. Use po(\"torch_loss\").")
      }
      if (is.null(md$optimizer)) {
        stopf("No optimizer configured in ModelDescriptor. Use po(\"torch_optimizer\").")
      }

      param_vals = self$param_set$get_values()

      learner = model_descriptor_to_learner(md)

      # TODO: Maybe we want the learner and the pipeop to actually share the paramset by reference.
      # If we do this we need to write a custom clone function.
      # While it is not efficient, the current solution works.
      learner$param_set$set_values(.values = param_vals)
      # in case something goes wrong during training we still set the state.
      on.exit({self$state = learner}, add = TRUE)
      learner$train(md$task)
      self$state = learner
      list(NULL)
    },
    .predict = function(inputs) {
      # This is copied from mlr3pipelines (PipeOpLearner)
      task = inputs[[1]]
      list(self$state$predict(task))
    },
    .task_type = NULL
  )
)

#' @title Torch Classification Model
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model_classif
#' @format [`R6Class`] object inheriting from [`PipeOpTorchModel`] / [`PipeOp`].
#'
#' @description
#' Builds a mlr3 Classification Torch Learner from its Input.
#' The default optimizer is adam an the default loss is cross entropy.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchModelClassif)`
#' * `r roxy_param_id("torch_model_classif")`
#' * `r roxy_param_param_vals()`
#'
#' @inheritSection mlr_pipeops_torch_model Input and Output Channels
#' @inheritSection mlr_pipeops_torch_model State
#' @template paramset_torchlearner
#' @inheritSection mlr_pipeops_torch_model Fields
#' @inheritSection mlr_pipeops_torch_model Methods
#' @inheritSection mlr_pipeops_torch_model Internals
#' @family PipeOpTorch
#' @export
PipeOpTorchModelClassif = R6Class("PipeOpTorchModelClassif",
  inherit = PipeOpTorchModel,
  public = list(
    initialize = function(id = "torch_model_classif", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        task_type = "classif"
      )
    }
  )
)

#' @title Torch Regression Model
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model_regr
#' @format `r roxy_format(PipeOpTorchModelRegr)`
#'
#' @description
#' Builds a Regression Torch Learner from its Input.
#' The default optimizer is adam an the default loss is mean-square error.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchModelRegr)`
#' * `r roxy_param_id("torch_model_regr")`
#' * `r roxy_param_param_vals()`
#'
#' @inheritSection mlr_pipeops_torch_model Input and Output Channels
#' @inheritSection mlr_pipeops_torch_model State
#' @template paramset_torchlearner
#' @inheritSection mlr_pipeops_torch_model Fields
#' @inheritSection mlr_pipeops_torch_model Methods
#' @inheritSection mlr_pipeops_torch_model Internals
#' @family PipeOpTorch
#' @export
PipeOpTorchModelRegr = R6Class("PipeOpTorchModelRegr",
  inherit = PipeOpTorchModel,
  public = list(
    initialize = function(id = "torch_model_regr", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        task_type = "regr"
      )
    }
  )
)

#' @include zzz.R
register_po("torch_model_regr", PipeOpTorchModelRegr)
register_po("torch_model_classif", PipeOpTorchModelClassif)
