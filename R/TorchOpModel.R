#' @export
TorchOpModel = R6Class("TorchOpModel",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "model", param_vals = list()) {
      param_set = ps(
        criterion = p_uty(tags = "train"),
        optimizer = p_uty(tags = "train"),
        optimizer_args = p_uty(tags = "train"),
        criterion_args = p_uty(tags = "train"),
        n_epochs = p_int(default = 0L, tags = "train", lower = 0L),
        device = p_fct(tags = c("train", "predict"), levels = c("cpu", "cuda"), default = "cpu"),
        batch_size = p_int(tags = c("train", "predict"), lower = 1L)
      )
      param_set$values$n_epochs = 0L
      output = data.table(name = "output", train = "Task", predict = "Prediction")

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
      inputs = inputs[[1L]]
      if (is.null(self$state$learner)) {
        task_type = inputs[["task"]]$task_type
        self$state$learner = switch(task_type,
          classif = LearnerClassifTorch$new(),
          regr = LearnerRegrTorch$new(),
          stopf("Invalid task type %s.", task_type)
        )
        self$state$learner$param_set$values = self$param_set$values
        self$state$learner$param_set$values$architecture = inputs[["architecture"]]
        self$state$learner$train(inputs[["task"]])
        list(task = inputs[["task"]])
      }
    },
    .predict = function(inputs) {
      prediction = self$state$learner$predict(inputs[["task"]])
      list(output = prediction)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("model", value = TorchOpModel)
