#' @title PipeOp Torch Model
#'
#' @name mlr_pipeops_torch_model
#'
#' @description
#' Builds a Torch Learner from a [`ModelDescriptor`] and trains it with the given parameter specification.
#' The task type must be specified during construction.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"` that takes in `ModelDescriptor` during traing and a `Task` of the specified
#' `task_type` during prediction.
#' The output is `NULL` during training and a `Prediction` of given `task_type` during prediction.
#'
#' @section State:
#' A trained [`LearnerTorchModel`].
#'
#' @template paramset_torchlearner
#'
#' @section Internals:
#' A [`LearnerTorchModel`] is created by calling [`model_descriptor_to_learner()`] on the
#' provided [`ModelDescriptor`] that is received through the input channel.
#' Then the parameters are set according to the parameters specified in `PipeOpTorchModel` and
#' its '$train()` method is called on the [`Task`] stored in the [`ModelDescriptor`].
#'
#' @family PipeOps
#' @export
PipeOpTorchModel = R6Class("PipeOpTorchModel",
  inherit = PipeOp,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param task_type (`character(1)`)\cr
    #'   The task type of the model.
    initialize = function(task_type, id = "torch_model", param_vals = list()) {
      # TODO: Add properties argument
      private$.task_type = assert_choice(task_type, c("classif", "regr"))
      param_set = paramset_torchlearner(task_type)
      input = data.table(
        name = "input",
        train = "ModelDescriptor",
        predict = mlr_reflections$task_types[private$.task_type, task]
      )
      output = data.table(
        name = "output",
        train = "NULL",
        predict = mlr_reflections$task_types[private$.task_type, prediction]
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
    .task_type = NULL,
    .additional_phash_input = function() {
      private$.task_type
    }
  )
)


#' @title PipeOp Torch Classifier
#'
#' @name mlr_pipeops_torch_model_classif
#'
#' @description
#' Builds a torch classifier and trains it.
#'
#' @inheritSection mlr_pipeops_torch_model Input and Output Channels
#' @inheritSection mlr_pipeops_torch_model State
#' @section Parameters: See [`LearnerTorch`]
#' @inheritSection mlr_pipeops_torch_model Internals
#' @family PipeOps
#' @export
#' @examplesIf torch::torch_is_installed()
#' @examples
#' # simple logistic regression
#'
#' # configure the model descriptor
#' md = as_graph(po("torch_ingress_num") %>>%
#'   po("nn_head") %>>%
#'   po("torch_loss", "cross_entropy") %>>%
#'   po("torch_optimizer", "adam"))$train(tsk("iris"))[[1L]]
#'
#' print(md)
#'
#' # build the learner from the model descriptor and train it
#' po_model = po("torch_model_classif", batch_size = 50, epochs = 1)
#' po_model$train(list(md))
#' po_model$state
PipeOpTorchModelClassif = R6Class("PipeOpTorchModelClassif",
  inherit = PipeOpTorchModel,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "torch_model_classif", param_vals = list()) {
      # TODO: Add properties argument
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
#' @name mlr_pipeops_torch_model_regr
#'
#' @description
#' Builds a torch regression model and trains it.
#'
#' @inheritSection mlr_pipeops_torch_model Input and Output Channels
#' @inheritSection mlr_pipeops_torch_model State
#' @section Parameters: See [`LearnerTorch`]
#' @inheritSection mlr_pipeops_torch_model Internals
#' @family PipeOps
#' @export
#' @examplesIf torch::torch_is_installed()
#' @examples
#' # simple linear regression
#'
#' # build the model descriptor
#' md = as_graph(po("torch_ingress_num") %>>%
#'   po("nn_head") %>>%
#'   po("torch_loss", "mse") %>>%
#'   po("torch_optimizer", "adam"))$train(tsk("mtcars"))[[1L]]
#'
#' print(md)
#'
#' # build the learner from the model descriptor and train it
#' po_model = po("torch_model_regr", batch_size = 20, epochs = 1)
#' po_model$train(list(md))
#' po_model$state
PipeOpTorchModelRegr = R6Class("PipeOpTorchModelRegr",
  inherit = PipeOpTorchModel,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "torch_model_regr", param_vals = list()) {
      # TODO: Add properties argument
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
