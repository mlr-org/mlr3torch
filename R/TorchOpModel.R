#' @title Builds a mlr3 Torch Learner from its input
#' @description Use TorchOpModeLClassif or TorchOpModelRegr for the respective task type.
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' criterion, ...) and afterwards trains the network.
#' It's parameterset is identical to LearnerClassifTorch and LearnerRegrTorch respectively.
#' @details TorchOpModelClassif and TorchOpModelRegr inherit from this R6 Class.
#' @export
TorchOpModel = R6Class("TorchOpModel",
  inherit = TorchOp,
  public = list(
    initialize = function(id, param_vals, .task_type, .optimizer) {
      assert_true(.optimizer %in% torch_reflections$optimizer)
      private$.optimizer = .optimizer
      private$.task_type = .task_type
      param_set = make_standard_paramset(.task_type, .optimizer, architecture = TRUE)
      param_set$values$epochs = 0L
      input = data.table(name = "input", train = "ModelArgs", predict = "Task")
      output = data.table(name = "output", train = "NULL", predict = "Prediction")

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        output = output
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      input = inputs$input
      learner = switch(private$.task_type,
        classif = LearnerClassifTorch$new(id = self$id, .optimizer = private$.optimizer),
        regr = LearnerRegrTorch$new(id = self$id, .optimizer = private$.optimizer),
        stopf("Unsupported task type '%s'.", private$.task_type)
      )
      # TODO: maybe the learner and the TorchOp should actually share the param-set?
      pars = self$param_set$get_values(tags = "train")
      if (!is.null(pars$architecture)) {
        warningf("Parameter 'architecture' was set, but is overwritten by ModelInput.")
      }
      pars$architecture = input$architecture
      learner$param_set$values = pars

      private$.learner = learner
      on.exit({
        private$.learner$state = NULL
      })
      self$state = private$.learner$train(input$task)$state

      list(NULL)
    },
    .predict = function(inputs) {
      # This is copied from mlr3pipelines (PipeOpLearner)
      on.exit({
        private$.learner$state = NULL
      })
      task = inputs[[1]]
      private$.learner$state = self$state
      list(private$.learner$predict(task))
    },
    .task_type = NULL,
    .learner = NULL,
    .optimizer = NULL
  )
)

#' @export
TorchOpModelClassif = R6Class(
  inherit = TorchOpModel,
  public = list(
    initialize = function(id = "model", param_vals = list(), .optimizer) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .task_type = "classif",
        .optimizer = .optimizer
      )
    }
  )
)

#' @export
TorchOpModelRegr = R6Class(
  inherit = TorchOpModel,
  public = list(
    initialize = function(id = "model", param_vals = list(), .optimizer) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .task_type = "regr",
        .optimizer = .optimizer
      )
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("model.regr", value = TorchOpModelRegr)

#' @include mlr_torchops.R
mlr_torchops$add("model.classif", value = TorchOpModelClassif)
