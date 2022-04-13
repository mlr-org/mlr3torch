#' TorchOpModelClassif and TorchOpModelRegr inherit from this R6 Class.
#' @export
TorchOpModel = R6Class("TorchOpModel",
  inherit = TorchOp,
  public = list(
    initialize = function(id, param_vals, .task_type, .optimizer) {
      private$.task_type = .task_type
      param_set = make_standard_paramset(.task_type, .optimizer)
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
        classif = LearnerClassifTorch$new(id = self$id),
        regr = LearnerRegrTorch$new(id = self$id),
        stopf("Unsupported task type '%s'.", private$.task_type)
      )
      pars = self$param_set$get_values(tags = "train")
      pars_piped = list(
        architecture = input$architecture,
        optimizer = input$optimizer,
        optim_args = input$optim_args,
        criterion = input$criterion,
        crit_args = input$crit_args
      )
      pars_piped = Filter(function(x) !is.null(x), pars_piped)
      pars = insert_named(pars, pars_piped)
      learner$param_set$values = pars

      private$.learner = learner
      on.exit({
        private$.learner$state = NULL
      })
      self$state = private$.learner$train(input$task)$state

      list(NULL)
    },
    .predict = function(inputs) {
      on.exit({
        private$.learner$state = NULL
      })
      task = inputs[[1]]
      private$.learner$state = self$state
      list(private$.learner$predict(task))
    },
    .task_type = NULL,
    .learner = NULL
  )
)

TorchOpModelClassif = R6Class(
  inherit = TorchOpModel,
  public = list(
    initialize = function(id = "model.classif", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .task_type = "classif"
      )
    }
  )
)

#' @export
TorchOpModelRegr = R6Class(
  inherit = TorchOpModel,
  public = list(
    initialize = function(id = "model.regr", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .task_type = "regr"
      )
    }
  )
)


#' @include mlr_torchops.R
mlr_torchops$add("model", value = TorchOpModel)

#' @include mlr_torchops.R
mlr_torchops$add("model.regr", value = TorchOpModelRegr)

#' @include mlr_torchops.R
mlr_torchops$add("model.classif", value = TorchOpModelClassif)
