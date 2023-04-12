#' @title PipeOp Torch Callbacks
#'
#' @usage NULL
#' @name mlr_pipeops_torch_callbacks
#' @format `r roxy_pipeop_torch_format(PipeOpTorchCallbacks)`
#'
#' @description
#' Configures the callbacks of a deep learning model.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchCallbacks)`
#' * `callbacks` :: `list` of [`TorchCallback`]s\cr
#'   The callbacks. Must have unique ids. Default is `list()`.
#' * `r roxy_param_id("torch_callbacks")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' The parameters are defined dynamically from the callbacks, where the id of the respective callbacks is the
#' respective set id.
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch, model_configuration
#' @export
#' @examples
# TODO:
PipeOpTorchCallbacks = R6Class("PipeOpTorchCallbacks", 
  inherit = PipeOp,
  public = list(
    initialize = function(callbacks = list(), id = "torch_callbacks", param_vals = list()) {
      private$.callbacks = assert_torch_callbacks(as_torch_callbacks(callbacks))
      assert_names(ids(private$.callbacks), type = "unique")
      walk(private$.callbacks, function(cb) {
        cb$param_set$set_id = cb$id
      })
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = invoke(ParamSetCollection$new, sets = map(private$.callbacks, "param_set")),
        param_vals = param_vals,
        input = input,
        output = output,
        packages = Reduce(union, map(private$.callbacks, "packages")) %??% character(0)
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      new_callbacks = map(private$.callbacks, function(cb) cb$clone(deep = TRUE))
      previous_callbacks = inputs[[1L]]$callbacks
      # Attention: What happens if both lists have length 1?
      assert_names(c(ids(new_callbacks), ids(previous)))
      if (is.null(previous_callbacks)) {
        callbacks = new_callbacks
      } else {
        callbacks = c(prvi)
      }

      callbacks = c(previous_)
      assert_names(c())
      assert_true(is.null(inputs[[1L]]#ca``))
      inputs[[1]]$ca
      return(inputs)
    }
    .callbacks = NULL, 
  )
)

#' @include zzz.R
register_po("torch_callbacks", PipeOpTorchCallbacks)
