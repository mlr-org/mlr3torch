#' @title Torch Model
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model
#' @format [`R6Class`] object inheriting from [`PipeOp`].
#'
#' @description
#' Builds a mlr3 Torch Learner from its Input.
#' Use [`TorchOpModelClassif`] or [`TorchOpModelRegr`] for the respective task type.
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' loss, ...) and afterwards trains the network.
#' It's parameterset is identical to [`LearnerClassifTorch`] and [`LearnerRegrTorch`] respectively.
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
#' A trained torch [`Learner`].
#'
#' @template paramset_torchlearner
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#'
#' @section Internals:
#' First a [`nn_graph`] is created by calling [`model_descriptor_to_module()`] and than a `LearnerClassifTorchModel`
#' is created from the `module`, is initialized for the specification provided in the input [`ModelDescriptor`].
#' Then the parameters are set according to the parameters specified in `PipeOpTorchModel` and its '$train()` method
#' is called on the [`Task`] provided through the [`ModelDescriptor`].
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchModel = R6Class("PipeOpTorchModel",
  inherit = PipeOp,
  public = list(
    initialize = function(id = "torch_model", param_vals = list(), task_type) {
      private$.task_type = assert_choice(task_type, c("classif", "regr"))
      param_set = paramset_torchlearner()
      input = data.table(name = "input", train = "ModelDescriptor", predict = mlr_reflections$task_types["classif", task])
      output = data.table(name = "output", train = "NULL", predict = mlr_reflections$task_types["classif", prediction])

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
      param_vals = self$param_set$get_values(tags = "train")

      class = switch(private$.task_type,
        regr = LearnerRegrTorchAbstract,
        classif = LearnerClassifTorchModel
      )

      network = model_descriptor_to_module(md, list(md$.pointer))
      network$reset_parameters()

      learner = class$new(
        network = network,
        ingress_tokens = md$ingress,
        optimizer = md$optimizer,
        loss = md$loss,
        packages = md$graph$packages
      )

      learner$param_set$values = insert_named(learner$param_set$values, param_vals)

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
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' loss, ...) and afterwards trains the network.
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
#' @format [`R6Class`] object inheriting from [`PipeOpTorchModel`] / [`PipeOp`].
#'
#' @description
#' Builds a mlr3 Regrssion Torch Learner from its Input.
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' loss, ...) and afterwards trains the network.
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
register_po("torch_model", PipeOpTorchModel)
register_po("torch_model_regr", PipeOpTorchModelRegr)
register_po("torch_model_classif", PipeOpTorchModelClassif)
