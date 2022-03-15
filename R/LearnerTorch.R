LearnerTorch = R6Class("DeepLearner",
  inherit = Learner,
  public = list(
    initialize = function(task_type, predict_types, param_set, properties) {
      super$initialize(
        id = sprintf("%s.torch", task_type),
        feature_types = c("numeric", "integer", "factor", "ordered"),
        predict_type = predict_types,
        packages = "torch",
        param_set = param_set,
        properties = properties,
        man = "mlr3torch::mlr_learners_deep_learner",
        task_type = task_type
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
        train_model(
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
      pars = self$param_set$get_values(tags = "predict")
      response = predict_torch(self$state$model, task, pars$device, pars$batch_size)
      response
      # predictions = predict_from_model(model, task)
      # list(response = predictions)
      # newdata = task$data(cols = task$feature_names)
      # response = invoke(predict, self$model, newdata = newdata, .opts = allow_partial_matching)
      # list(response = unname(response))
    }
  )
)
