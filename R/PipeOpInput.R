#' @export
PipeOpInput = R6Class("PipeOpInput",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "input", param_vals = list()) {
      param_set = ps()

      input = data.table(name = "input", train = "*", predict = "*")
      output = data.table(
        name = c("task", "architecture"),
        train = c("Task", "Architecture"),
        predict = c("Task", "*")
      )

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
      architecture = Architecture$new()
      task = inputs[["input"]]
      output = list(task = task, architecture = architecture)
      # self$state = "trained"
      return(output)
    },
    .predict = function(inputs) {
      task = inputs[["input"]]
      architecture = Architecture$new()
      output = list(task = task, architecture = NULL)
      return(output)
    }
  )
)

if (FALSE) {
  devtools::load_all(".")
  pipeop = PipeOpInput$new()
  task = tsk("iris")
  output_train = pipeop$train(list(task))
}
