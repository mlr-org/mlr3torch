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
#'   [PipeOpTorch]'s `.shapes_out()` method for details on the parameters,
#'   and [PipeOpTaskPreprocTorch] for details on how the shapes are inferred when
#'   this parameter is NULL.
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
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_fn", param_vals = list()) {
      param_set = ps(
        fn = p_uty(tags = c("train", "required"), custom_check = check_function),
        shapes_out = p_uty(tags = "train", custom_check = function(input) {
          check_function(input, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE)
        })
      )

      super$initialize(
        id = id,
        param_set = c(param_set, inferps(param_vals$fn)),
        param_vals = param_vals,
        module_generator = NULL
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (!is.null(param_vals$shapes_out)) {
        new_shapes = param_vals$shapes_out(shapes_in = shapes_in, param_vals = param_vals, task = task)
        assert_list(new_shapes, types = "integer", any.missing = TRUE)
        assert_subset(names(new_shapes), self$output$name, empty.ok = FALSE)
        return(new_shapes)
      }

      infer_shapes(shapes_in = shapes_in, param_vals = param_vals, output_names = self$output$name, fn = param_vals$fn, rowwise = FALSE, id = self$id)
    },
    .make_module = function(shapes_in, param_vals, task) {
      nn_module("nn_fn",
        initialize = function(fn) {
          self$fn = fn
          self$args = param_vals[!(names(param_vals) %in% c("fn", "shapes_out"))]
        },
        forward = function(x) {
          return(invoke(self$fn, x, .args = self$args))
        }
      )(param_vals$fn)
    }
  )
)

#' @include aaa.R
register_po("nn_fn", PipeOpTorchFn)
