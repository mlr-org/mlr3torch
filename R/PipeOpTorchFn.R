#' @title Custom Function
#' @name mlr_pipeops_nn_fn
#'
#' @description
#' Applies a user-supplied function to a tensor.
#'
#' @section Parameters:
#' By default, these are inferred as all but the first arguments of the function `fn`.
#' It is also possible to specify these more explicitly via the `param_set` constructor argument.
#'
#' @templateVar id nn_fn
#' @template pipeop_torch_channels_default
#'
#' @examplesIf torch::torch_is_installed()
#' custom_fn =  function(x, a) x / a
#' obj = po("nn_fn", fn = custom_fn, a = 2)
#' obj$param_set
#'
#' graph = po("torch_ingress_ltnsr") %>>% obj
#'
#' task = tsk("lazy_iris")$filter(1)
#' tnsr = materialize(task$data()$x)[[1]]
#'
#' md_trained = graph$train(task)
#' trained = md_trained[[1]]$graph$train(tnsr)
#'
#' trained[[1]]
#'
#' custom_fn(tnsr)
#' @export
PipeOpTorchFn = R6Class("PipeOpTorchFn",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [`R6`][R6::R6Class] class.
    #' @param fn (`function`)\cr
    #' The function to be applied. Takes a `torch` tensor as first argument and returns a `torch` tensor.
    #' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
    #' A ParamSet wrapping the arguments to `fn`.
    #' If omitted, then the ParamSet for this PipeOp will be inferred from the function signature.
    #' @param shapes_out (`function` or `NULL`)\cr
    #' A function that computes the output shapes of the `fn`. See
    #' [PipeOpTorch]'s `.shapes_out()` method for details on the parameters,
    #' and [PipeOpTaskPreprocTorch] for details on how the shapes are inferred when
    #' this parameter is `NULL`.
    #' @template params_pipelines
    initialize = function(fn, id = "nn_fn", param_vals = list(), param_set = NULL, shapes_out = NULL) {
      private$.fn = assert_function(fn)

      if (is.null(param_set)) {
        param_set = inferps(private$.fn, ignore = formalArgs(private$.fn)[1])
      } else {
        assert_param_set(param_set)
        assert_subset(param_set$ids(), formalArgs(private$.fn))
      }

      if (!is.null(shapes_out)) {
        private$.shapes_out_fn = assert_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE)
      }

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = NULL
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (!is.null(private$.shapes_out_fn)) {
        new_shapes = private$.shapes_out_fn(shapes_in = shapes_in, param_vals = param_vals, task = task)
        assert_list(new_shapes, types = "integer", any.missing = TRUE)
        assert_subset(names(new_shapes), self$output$name, empty.ok = FALSE)
        return(new_shapes)
      }

      infer_shapes(shapes_in = shapes_in, param_vals = param_vals, output_names = self$output$name, fn = private$.fn, rowwise = FALSE, id = self$id)
    },
    .make_module = function(shapes_in, param_vals, task) {
      nn_module("nn_fn",
        initialize = function(fn, param_vals) {
          self$fn = fn
          self$args = param_vals
        },
        forward = function(x) {
          invoke(self$fn, .args = c(list(x), self$args))
        }
      )(private$.fn, param_vals)
    },
    .fn = NULL,
    .shapes_out_fn = NULL,
    .additional_phash_input = function() {
      hash_input(private$.fn)
    }
  )
)

#' @include aaa.R utils.R
register_po("nn_fn", PipeOpTorchFn, metainf = list(fn = identity))
