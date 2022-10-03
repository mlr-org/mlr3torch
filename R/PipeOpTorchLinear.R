#' @inherit torch::nn_linear title
#' @inherit torch::nn_linear description
#'
#' @name mlr_pipeops_nn_linear
#' @format [`R6Class`] object inheriting from [`PipeOpTorch`]/[`PipeOp`].
#'
#' @section Calls:
#' Calls [torch::nn_linear()] when trained.
#'
#' @section Input and Output Channels:
#' See [PipeOpTorch] for a description.
#' Input and output channels take
#'
#' The output is the input [`Task`][mlr3::Task] with all affected numeric features replaced by their
#' non-negative components.
#'
#' @inheritSection PipeOpTorch
#' The `$state` is a named `list` with the `$state` elements inherited from [`PipeOpTaskPreproc`],
#' as well as the elements of the object returned by [`nmf()`][NMF::nmf].
#'
#' @section Parameters:
#'
#' **Available**
#'
#' * `out_features` :: `integer(1)`\cr
#'   Size of each output sample.
#' * `bias` :: `logical(1)`\cr
#'   If set to `FALSE`, the layer will not learn an additive bias. Default: `TRUE`
#'
#' **Inferred**
#'
#' * `in_features` :: `integer(1)`\cr
#'   Last dimension of the input shape.
#'
#' @inheritSection torch::nn_linear Shape
#' @inheritSection torch::nn_linear Attributes
#' @inheritSection torch::nn_linear references
#'
#' @section Methods:
#' Only methods inherited from [`PipeOpTorch`]/[`PipeOp`].
#'
#' @examples
#' # po
#' obj = po("nn_linear", out_features = 10)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
#'
#' # pot
#' obj = pot("linear", out_features = 10)
#' obj$id
#' obj$module_generator
#'
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchLinear = R6Class("PipeOpTorchLinear",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_linear
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals) c(param_vals, list(in_features = tail(shapes_in[[1]], 1))),
    .shapes_out = function(shapes_in, param_vals) list(c(head(shapes_in[[1]], -1), param_vals$out_features))
  )
)

#' @include zzz.R
register_po("nn_linear", PipeOpTorchLinear)

