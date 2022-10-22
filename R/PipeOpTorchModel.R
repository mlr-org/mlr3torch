#' @title Builds a mlr3 Torch Learner from its Input
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model
#' @format [`R6Class`] inheriting from [`PipeOp`].
#'
#' @description Use TorchOpModeLClassif or TorchOpModelRegr for the respective task type.
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' loss, ...) and afterwards trains the network.
#' It's parameterset is identical to LearnerClassifTorch and LearnerRegrTorch respectively.
#'
#' @section Construction:
#' ```
#' PipeOpTorchModel$new(id, param_vals, task_type)
#' ```
#' `r roxy_param_id()`
#' `r roxy_param_param_set()`
#' * `task_type` :: `character(1)`\cr
#'   The task type of the model. See `mlr_reflections$task_types` for available options.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"` that takes in `ModelDescriptor` during traing and a `Task` of the specified
#' `task_type` during prediction.
#' The output is `NULL` during training and a `Prediction` of given `task_type` during prediction.
#'
#' @section State:
#' A trained torch [`Learner`].
#'
#' @section Parameters:
#' * `batch_size` :: (`integer(1)`)\cr
#'   The batch size.
#' * `epochs` :: `integer(1)`\cr
#'   The number of epochs.
#' * `device` :: `character(1)`\cr
#'   The device. One of `"auto"`, `"cpu"`, or `"cuda"`.
#' * `measures_train` :: `list()` of [`Measure`]s.
#'   Measures to be evaluated during training.
#' * `measures_valid` :: `list()` of [`Measure`]s.
#'   Measures to be evaluated during validation.
#' * `augmentation` :: ??
#'  TODO:
#' * `callbacks` :: (list of) `CallbackTorch`\cr
#'   The callbacks to .
#' * `drop_last` :: `logical(1)`\cr
#'   Whether to drop the last batch in each epoch during training. Default is `FALSE`.
#' * `num_threads` :: `integer(1)`\cr
#'   The number of threads (if `device` is `"cpu"`). Default is 1.
#' * `shuffle` :: `logical(1)`\cr
#'   Whether to shuffle the instances in the dataset. Default is `TRUE`.
#'
#' @section Internals:
#' First a [`nn_graph`] is created by calling [`model_descriptor_to_module()`] and than a `LearnerClassifTorchModel`
#' is created from the `module`, is initialized for the specification provided in the input [`ModelDescriptor`].
#' Then the parameters are set according to the parameters specified in `PipeOpTorchModel` and its '$train()` method
#' is called on the [`Task`] provided through the [`ModelDescriptor`].
#'
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#'
#' @section Methods:
#' Only Methods inherited from [`PipeOp`].
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchModel = R6Class("PipeOpTorchModel",
  inherit = PipeOp,
  public = list(
    initialize = function(id, param_vals, task_type) {
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
      param_vals = self$param_set$get_values()

      class = switch(private$.task_type,
        regr = LearnerRegrTorchModel,
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

#' @title Builds a mlr3 Torch Classification Learner from its Input
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model_classif
#' @format [`R6Class`] inheriting from [`PipeOp`].
#'
#' @description
#' Create and train [`LearnerClassifTorchModel`] specified by a [`ModelDescriptor`].
#'
#' @section Construction:
#' ```
#' PipeOpTorchModelClassif$new(id = "torch_model.classif", param_vals)
#' ```
#' `r roxy_param_id()`
#' `r roxy_param_param_set()`
#'
#' @inheritSection mlr_pipeops_torch Input and Output Channels
#'
#' @section State:
#' A trained [`LearnerClassifTorchModel`].
#'
#' @inheritSection mlr_pipeops_torch_model Parameters
#'
#' @inheritSection mlr_pipeops_torch_model Internals
#'
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#'
#' @section Methods:
#' Only Methods inherited from [`PipeOp`].
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchModelClassif = R6Class("PipeOpTorchModelClassif",
  inherit = PipeOpTorchModel,
  public = list(
    initialize = function(id = "torch_model.classif", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        task_type = "classif"
      )
    }
  )
)

#' @title Builds a mlr3 Torch Regression Learner from its Input
#'
#' @usage NULL
#' @name mlr_pipeops_torch_model_regr
#' @format [`R6Class`] inheriting from [`PipeOp`].
#'
#' @description
#' Create and train [`LearnerRegrTorchModel`] specified by a [`ModelDescriptor`].
#'
#' @section Construction:
#' ```
#' PipeOpTorchModelReg$new(id = "torch_model.regr", param_vals = list())
#' ```
#' `r roxy_param_id()`
#' `r roxy_param_param_set()`
#'
#' @inheritSection mlr_pipeops_torch Input and Output Channels
#'
#' @section State:
#' A trained [`LearnerClassifTorchModel`].
#'
#' @inheritSection mlr_pipeops_torch_model Parameters
#'
#' @inheritSection mlr_pipeops_torch_model Internals
#'
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#'
#' @section Methods:
#' Only Methods inherited from [`PipeOp`].
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchModelRegr = R6Class("PipeOpTorchModelRegr",
  inherit = PipeOpTorchModel,
  public = list(
    initialize = function(id = "torch_model.regr", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        task_type = "regr"
      )
    }
  )
)

#' @include zzz.R
register_po("torch_model.regr", PipeOpTorchModelRegr)
register_po("torch_model.classif", PipeOpTorchModelClassif)
