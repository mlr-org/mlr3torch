#' @title Callback Configuration
#'
#' @name mlr_pipeops_torch_callbacks
#'
#' @description
#' Configures the callbacks of a deep learning model.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"` and one output channel `"output"`.
#' During *training*, the channels are of class [`ModelDescriptor`].
#' During *prediction*, the channels are of class [`Task`][mlr3::Task].
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' The parameters are defined dynamically from the callbacks, where the id of the respective callbacks is the
#' respective set id.
#' @section Internals:
#' During training the callbacks are cloned and added to the [`ModelDescriptor`].
#' @family Model Configuration
#' @family PipeOp
#' @export
#' @examplesIf torch::torch_is_installed()
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
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param callbacks (`list` of [`TorchCallback`]s) \cr
    #'   The callbacks (or something convertible via [`as_torch_callbacks()`]).
    #'   Must have unique ids.
    #'   All callbacks are cloned during construction.
    initialize = function(callbacks = list(), id = "torch_callbacks", param_vals = list()) {
      private$.callbacks = as_torch_callbacks(callbacks, clone = TRUE)
      cbids = ids(private$.callbacks)
      assert_names(cbids, type = "unique")
      private$.callbacks = set_names(private$.callbacks, cbids)
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = alist(ParamSetCollection$new(sets = map(private$.callbacks, "param_set"))),
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
        # TODO: Is this necessary?
        callbacks = map(private$.callbacks, function(cb) cb$clone(deep = TRUE))
        return(callbacks)
      }
      super$deep_clone(name, value)
    },
    .additional_phash_input = function() {
      map(private$.callbacks, "phash")
    }

  )
)

#' @include zzz.R
register_po("torch_callbacks", PipeOpTorchCallbacks)
