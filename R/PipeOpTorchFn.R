#' @title Custom Function
#' @name mlr_pipeops_nn_fn
#' 
#' @description
#' Applies a user-supplied function to a tensor.

#' @section Parameters:
#' * `fn` :: `function`\cr
#'   The function to apply. Takes a `torch` tensor as its first argument and returns a `torch` tensor.
#' * `shapes_out` :: `function`\cr
#'   (`list()`, `list()`, [`Task`][mlr3::Task] or `NULL`) -> named `list()`\cr
#'   A function that computes the output shapes of the `fn`. See 
#'   [PipeOpTorch]'s `.shapes_out()` method for details.
#' @templateVar id nn_fn
#' @template pipeop_torch_channels_default
#'
#' @examplesIf torch::torch_is_installed()
#' custom_fn =  function(x) x / 2
#' po = po("nn_fn", param_vals = list(fn = custom_fn))
#' po$param_set
#' 
#' graph = po("torch_ingress_ltnsr") %>>% po
#'
#' ds_len = 2
#' data_tnsr = torch_randn(c(ds_len, 2, 2))
#' 
#' ds_gen = dataset(name = "dummy", 
#'   initialize = function(x) self$x = x, 
#'   .getitem = function(i) list(x = self$x[i, ]), 
#'   .length = function() dim(self$x)[1]
#' )
#' 
#' ds = ds_gen(data_tnsr)
#' 
#' dd = as_data_descriptor(ds, list(x = c(NA, 2, 2)))
#' lt = lazy_tensor(dd)
#' task = as_task_classif(data.table(y = factor(c(0, 1)), x = lt), target = "y")
#'  
#' md_trained = graph$train(task)
#' trained = md_trained[[1]]$graph$train(data_tnsr)
#'
#' trained[[1]]
#' 
#' custom_fn(data_tnsr)
#' @export
PipeOpTorchFn = R6Class("PipeOpTorchFn",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_fn", param_vals = list()) {
      param_set = ps(
        fn = p_uty(tags = c("train", "required"), custom_check = check_function),
        shapes_out = p_uty(tags = "train", custom_check = function(input) check_function(input, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE))
      )

      ps_fn = inferps(param_vals$fn)

      super$initialize(
        id = id,
        param_set = c(param_set, ps_fn),
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
      tensor_out = tryCatch(mlr3misc::invoke(param_vals$fn, tensor_in),
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
      nn_module("nn_fn",
        initialize = function(fn) {
          self$fn = fn
        },
        forward = function(x) {
          return(self$fn(x))
        }
      )(param_vals$fn)
    }
  )
)

#' @include aaa.R
register_po("nn_fn", PipeOpTorchFn)