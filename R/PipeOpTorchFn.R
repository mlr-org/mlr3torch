#' @title Custom Function
#' @name mlr_pipeops_nn_fn
#' 
#' @description
#' Applies a user-supplied function to a tensor.

#' @section Parameters:
#' * `fn` :: `function`\cr
#'   The function to apply. Takes a `torch` tensor as its first argument and returns a `torch` tensor.
#'
#' @templateVar id nn_fn
#' @template pipeop_torch_channels_default
#'
#' @example
#' 
#' @export
PipeOpTorchFn = R6Class("PipeOpTorchFn",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_fn", param_vals = list()) {
      param_set = ps(
        fn = p_uty(tags = c("train", "required")),
        shapes_out = p_uty(tags = "train", custom_check = function(input) check_integer(input, null.ok = TRUE))
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = NULL
      )
    }
  ),
  private = list(
    .shapes_out =function(shapes_in, param_vals, task) {
      sin = shapes_in[["input"]]
      batch_dim = sin[1L]
      batchdim_is_unknown = is.na(batch_dim)
      if (batchdim_is_unknown) {
        sin[1] = 1L
      }
      tensor_in = mlr3misc::invoke(torch_empty, .args = sin, device = torch_device("cpu"))
      tensor_out = tryCatch(mlr3misc::invoke(param_vals$fn, tensor_in, .args = param_vals),
        error = function(e) {
          stopf("Input shape '%s' is invalid for PipeOp with id '%s'.", shape_to_str(list(sin)), self$id)
        }
      )
      sout = dim(tensor_out)
      if (batchdim_is_unknown) {
        sout[1] = NA
      }

      list(sout)
    },
    .make_module = function(shapes_in, param_vals, task) {
      self$param_set$values$fn
    }
  )
)

#' @include aaa.R
register_po("nn_fn", PipeOpTorchFn)