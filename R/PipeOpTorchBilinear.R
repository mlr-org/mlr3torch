#' @title Bilinear Transformation
#' @inherit torch::nn_bilinear description
#' @section nn_module:
#' Calls [`torch::nn_bilinear()`] when trained.
#' The parameters `in1_features` and `in2_features` are inferred as the last dimension of the two
#' input shapes.
#' @section Parameters:
#' * `out_features` :: `integer(1)`\cr
#'   The dimension of the output.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias. Default is `TRUE`.
#'
#' @section Input and Output Channels:
#' Two input channels called `"input1"` and `"input2"` and one output channel called `"output"`.
#' Both inputs must have the same number of dimensions and agree in all of them but the last, which
#' becomes `in1_features` and `in2_features` respectively.
#' For an explanation see [`PipeOpTorch`].
#'
#' @templateVar id nn_bilinear
#' @templateVar param_vals out_features = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchBilinear = R6Class("PipeOpTorchBilinear",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_bilinear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(lower = 1L, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_bilinear,
        inname = c("input1", "input2")
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      assert_same_ndim(shapes_in, self$id)
      shape1 = shapes_in[[1L]]
      shape2 = shapes_in[[2L]]
      # the last dimensions become in1_features and in2_features, which the module needs to size its
      # weights, so neither of them may be unknown
      assert_known_dims(shape1, length(shape1), "the last dimension (which becomes 'in1_features')",
        self$id)
      assert_known_dims(shape2, length(shape2), "the last dimension (which becomes 'in2_features')",
        self$id)
      # torch requires the leading dimensions to be equal and does not broadcast them, so a known
      # mismatch is an error rather than something to be resolved at runtime
      lead1 = utils::head(shape1, -1L)
      lead2 = utils::head(shape2, -1L)
      mismatch = !is.na(lead1) & !is.na(lead2) & lead1 != lead2
      if (any(mismatch)) {
        stopf("PipeOp '%s' requires its two inputs to agree in all dimensions but the last, but the shapes %s and %s differ in dimension %i.", # nolint
          self$id, shape_to_str(shape1), shape_to_str(shape2), which(mismatch)[[1L]])
      }
      # a dimension that only one input knows is known for the output as well, since they must agree
      lead = ifelse(is.na(lead1), lead2, lead1)
      list(as.integer(c(lead, param_vals[["out_features"]])))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$in1_features = utils::tail(shapes_in[[1L]], 1L)
      param_vals$in2_features = utils::tail(shapes_in[[2L]], 1L)
      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_bilinear", PipeOpTorchBilinear)
