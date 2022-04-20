#' @include TorchOp.R
#' @export
TorchOpRepeat = R6Class(
  inherit = TorchOp,
  public = list(
    initialize = function(id = "repeat", param_vals = list()) {
      param_set = ps(
        times = p_int(default = NO_DEF, lower = 0L, tags = c("train", "required")),
        last = p_int(default = 1L, lower = 0L, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      params = self$param_set$get_values(tag = "train")
      rep = params[["rep"]]
      last = params[["last"]] %??% 1L
      if (!is.null(self$state)) {
        # architecture is already built
        return(list(task = inputs[["task"]], architecture = NULL))
      }
      task = inputs[["task"]]
      architecture = inputs[["architecture"]]
      assert(length(architecture$layers) >= last)
      layers = tail(architecture, last)
      for (i in seq_len(times)) {
        for (layer in layers) {
          architecture$add(layer[["operator"]], layer[["param_vals"]])
        }
      }
      architecture$add(private$.operator, self$param_set$get_values(tag = "train"))
      self$state = list()
      output = list(task = inputs[["task"]], architecture = architecture)
      return(output)

    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("repeat", TorchOpRepeat)
