LearnerRegrTorch = R6Class("LearnerRegrTorch",
  inherit = Learner,
  public = list(
    initialize = function() {
      super$initialize(
        id = "regr.torch",
        feature_types = c("numeric", "integer", "factor", "ordered"),
        predict_type = "response",
        packages = "torch",
        param_set = dl_paramset(),
        properties = c(),
        man = "mlr3torch::mlr_learners_deep_learner",
        task_type = "regr"
      )
    }
  ),
  private = list(
    .train = function(task) {
      if (!length(self$state)) {
        private$.build(task)
      }
      if (self$param_set$values$n_epochs > 0L) {
        assert(!is.null(self$state$optimizer))
        assert(!is.null(self$state$criterion))
        train_torch(
          model = self$state$model,
          task = task,
          optimizer = self$state$optimizer,
          criterion = self$state$criterion,
          n_epochs = self$param_set$values$n_epochs,
          batch_size = self$param_set$values$batch_size,
          device = self$param_set$values$device
        )
      }
      return(self$state$model)
    },

    .build = function(task) {
      pars = self$param_set$get_values(tag = "train")
      reduction = reduce_architecture(pars[["architecture"]], task)
      model = reduction[["model"]]
      self$state = list(
        model = model,
        optimizer = mlr3misc::invoke(pars[["optimizer"]], .args = pars[["optimizer_args"]],
          params = model$parameters
        ),
        criterion = mlr3misc::invoke(pars[["criterion"]], .args = pars[["criterion_args"]])
      )
    },

    .predict = function(task) {
      assert(task$task_type == "regr")
      params = self$param_set$get_values(tags = "predict")
      response = invoke(predict_torch, model = self$state$model, task = task, .args = params)
      return(response)
    }
  )
)
