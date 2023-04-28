#' @title Callback Configuration
#'
#' @usage NULL
#' @name mlr_pipeops_torch_callbacks
#' @format `r roxy_format(PipeOpTorchCallbacks)`
#'
#' @description
#' Configures the callbacks of a deep learning model.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchCallbacks)`
#' * `callbacks` :: `list` of [`TorchCallback`]s or `character()` or \cr
#'   The callbacks (or something convertible via [`as_torch_callbacks()`]).
#'   Must have unique ids. Default is `list()`.
#'   All callbacks are cloned during construction.
#' * `r roxy_param_id("torch_callbacks")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"` and one output channel `"output"`.
#' During *training*, the channels are of class [`ModelDescriptor`].
#' During *prediction*, the channels are of class [`Task`].
#'
#' @section State:
#' The state is set to an empty `list()`.
#'
#' @section Parameters:
#' The parameters are defined dynamically from the callbacks, where the id of the respective callbacks is the
#' respective set id.
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#' @section Methods:
#' Only methods inherited from [`PipeOp`].
#' @section Internals:
#' During training the callbacks are cloned and added to the [`ModelDescriptor`].
#' @family model_configuration
#' @export
#' @examples
#' po_cb = po("torch_callbacks", "checkpoint")
#' po_cb$param_set
#' mdin = po("torch_ingress_num")$train(list(tsk("iris")))
#' mdin[[1L]]$callbacks
#' mdout = po_cb$train(mdin)[[1L]]
#' mdout$callbacks
#' # Can be called again
#' po_cb1 = po("torch_callbacks", t_clbk("progress"))
#' mdout1 = po_cb1$train(list(mdout))[[1L]]
#' mdout1$callbacks
PipeOpTorchCallbacks = R6Class("PipeOpTorchCallbacks",
  inherit = PipeOp,
  public = list(
    initialize = function(callbacks = list(), id = "torch_callbacks", param_vals = list()) {
      private$.callbacks = as_torch_callbacks(callbacks, clone = TRUE)
      cbids = ids(private$.callbacks)
      assert_names(cbids, type = "unique")
      walk(private$.callbacks, function(cb) {
        cb$param_set$set_id = cb$id
        walk(cb$param_set$params, function(p) {
          p$tags = union(p$tags, "train")
        })
      })
      private$.callbacks = set_names(private$.callbacks, cbids)
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = alist(invoke(ParamSetCollection$new, sets = map(private$.callbacks, "param_set"))),
        param_vals = param_vals,
        input = input,
        output = output,
        packages = Reduce(union, map(private$.callbacks, "packages")) %??% character(0)
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      callbacks = c(
        map(private$.callbacks, function(cb) cb$clone(deep = TRUE)),
        as_torch_callbacks(inputs[[1L]]$callbacks)
      )
      cbids = ids(callbacks)
      if (!test_names(cbids, type = "unique")) {
        dups = cbids[duplicated(cbids)]
        stopf("Callbacks with IDs %s are already present.", paste0("'", dups, "'", collapse = ", "))
      }
      inputs[[1]]$callbacks = callbacks
      self$state = list()
      return(inputs)
    },
    .predict = function(inputs) inputs,
    .callbacks = NULL,
    deep_clone = function(name, value) {
      if (name == ".callbacks") {
        callbacks = map(private$.callbacks, function(cb) cb$clone(deep = TRUE))
        return(callbacks)
      }
      super$deep_clone(name, value)
    }
  )
)

#' @include zzz.R
register_po("torch_callbacks", PipeOpTorchCallbacks)
