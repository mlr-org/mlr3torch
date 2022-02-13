#' @title PipeOpDeepLearner
#' @description Wraps a DeepLearner
#' @details It builds a NeuralNetwork from a Architecture and then builds a DeepLearner from the
#' NeuralNetwork and the remaining parameters
#' @export
#' @include
PipeOpDeepLearner = R6::R6Class("PipeOpDeepLearner",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    optimizer = NULL,
    optim_args = NULL,
    criterion = NULL,
    criterion_args = NULL,
    train_args = NULL,
    initialize = function(id = "neural.network", param_vals = list()) {
      param_set = dl_paramset(network = FALSE)
      input = data.table(
        name = c("task", "architecture"),
        train = c("Task", "Architecture"),
        predict = c("Task", "*") # During predict the input in architecture should be NULL
      )
      output = data.table(name = "task", train = "Task", predict = "Task")
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
    .train = function(input) {
      task = input[["task"]]
      architecture = input[["architecture"]]
      pars = self$param_set$get_values(tag = "train")
      if (is.null(self$state)) {
        private$.build()
      }
      train_model(
        model = self$state$model,
        task = task,
        n_steps = self$p,
        optimizer = self$state$optimizer,
        criterion = self$state$criterion
      )
    },
    .build = function(input) {
      self$state$learner = DeepLearner$new()
      self$state$learner$param_set$values = c(
        self$param_set$get_values(tag = "train")
      )
    }
  )
)




if (FALSE) {
  lin = TorchOpLinear$new(param_vals = list(units = 10))
  deeplearner = TorchOpLinear$new()
  task = tsk("iris")

  devtools::load_all("~/mlr/mlr3pipelines")
  Graph$debug("train")
  graph = lin %>>% mod

  graph$train(task)
}



# Checkout all the params of optimizers implemented in torch
# library(torch)
#
# ns = getNamespace("torch")
# idxs = grep("optim", names(ns))
# names = names(ns)[idxs]
# f = function(x) get(x, env = ns)
# tops = map(names, f)
#
# params = unique(unlist(map(tops, formalArgs)))
#
#
##
