#' @export
PipeOpModel = R6::R6Class("PipeOpModel",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    optimizer = NULL,
    optim_args = NULL,
    criterion = NULL,
    criterion_args = NULL,
    train_args = NULL,
    initialize = function(id = "neural.network", param_vals = list()) {

      param_set = ps(
        optimizer = p_fct(levels = c("sgd", "rmsprop"), tags = "train"),
        weight_decay = p_dbl(0, 1, tags = "train"),
        momentum = p_dbl(0, 1, tags = "train"),
        criterion = p_fct(levels = c("nn_cross_entropy_loss"))
      )

      input = data.table(
        name = c("task", "architecture"),
        train = c("Task", "Architecture"),
        predict = c("Task", "*")
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
      if (is.null(self$state$learner)) {
        self$state$model = reduce_architecture(architecture, task)
      }
      train_model(
        model = self$state$model,
        task = task,
        train_args = self$train_args,
        optimizer = self$state$optimizer,
        criterion = self$state$criterion
      )
    },
    .build = function(input) {
      task = input[["task"]]
      architecture = input[["architecture"]]

      self$state$model = build_model(architecture, task)
      self$state$criterion = mlr3misc::invoke(get(sprintf("nn_%s", self$criterion),
        envir = getNamespace("torch")), .args = self$criterion_args)
      self$state$optimizer = mlr3misc::invoke(get(sprintf("optim_%s", self$criterion),
        envir = getNamespace("torch")), .args = self$criterion_args)
    }
  )
)

train_model = function(model, task, train_args, optimizer, criterion) {
  n_epochs = train_args[["n_epochs"]]
  batch_size = train_args[["batch_size"]]
  device = train_args[["device"]]

  dl = task$dataloader()
  dl$batch_size = batch_size

  for (epoch in seq_len(n_epochs)) {
    losses = list()
    coro::loop(for (batch in dl) {
      optimizer$zero_grad()
      output = model$forward(batch$x$to(device = device))
      loss = criterion(y_hat, batch$y$to(device = device))
      optimizer$step()
      losses = c(losses, loss$item())
    })
  }

}

reduce_architecture = function(architecture, task) {
  init = list(
    x = task$data$.getitem(1),
    network = NeuralNetwork$new()
  )
  f = function(lhs, rhs) {
    layer = rhs[["bob"]](lhs[["x"]], lhs[["network"]], rhs[["param_vals"]])
    output = list(
      x = with_no_grad(layer(x)),
      network = network$append(layer)
    )
    return(output)
  }
  output = Reduce(f, architecture$layers, init)
}


if (FALSE) {
  lin = PipeOpLinear$new(param_vals = list(units = 10))
  mod = PipeOpModel$new()
  task = tsk("iris")

  devtools::load_all("~/mlr/mlr3pipelines")
  Graph$debug("train")
  pipeline = lin %>>% mod

  pipeline$train(list(task))
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
#
# get(names(ns)[grep("optim", names(ns))], env = ns)# get(names(ns)[grep("optim", names(ns))], env = ns)# get(names(ns)[grep("optim", names(ns))], env = ns)# get(names(ns)[grep("optim", names(ns))], env = ns)
