TorchOpModel = R6Class("TorchOpModel",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "model", param_vals = list()) {
      param_set = ps(
        criterion = p_uty(tags = "train", custom_check = criterion_check, trafo = criterion_trafo),
        optimizer = p_uty(tags = "train", custom_check = optimizer_check, trafo = optimizer_trafo),
        optimizer_args = p_uty(tags = "train"),
        criterion_args = p_uty(tags = "train"),
        n_epochs = p_int(tags = "train", lower = 0L),
        device = p_fct(tags = c("train", "predict"), levels = c("cpu", "cuda"), default = "cpu"),
        batch_size = p_int(tags = c("train", "predict"), lower = 1L)
      )
      output = data.table(
        name = "network"
      )
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
      if (is.null(self$state$learner)) {
        task_type = inputs[["task"]]$task_type
        class = switch(task_type,
          classif = LearnerClassifClassif,
          regr = LearnerClassifRegr,
          stopf("Invalid task type %s.", task_type)
        )
        self$state$learner = invoke(class$new, architecture = inputs[["architecture"]],
          .args = self$param_set$get_values("train")
        )
      }

    }
  )
)
