#' @title Group Normalization
#' @inherit torch::nnf_group_norm description
#' @section nn_module:
#' Calls [`torch::nn_group_norm()`] when trained.
#' The parameter `num_channels` is inferred as the second dimension of the input shape.
#' @section Parameters:
#' * `num_groups` :: `integer(1)`\cr
#'   The number of groups to separate the channels into. Must divide the number of channels.
#'   Setting it to `1` normalizes over all channels at once (layer normalization), setting it to
#'   the number of channels normalizes each channel on its own (instance normalization).
#' * `eps` :: `numeric(1)`\cr
#'   A value added to the denominator for numerical stability. Default: `1e-5`.
#' * `affine` :: `logical(1)`\cr
#'   Whether to learn per-channel affine parameters initialized to `1` for weights and to `0` for
#'   biases. Default: `TRUE`.
#'
#' @templateVar id nn_group_norm
#' @templateVar param_vals num_groups = 1
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchGroupNorm = R6Class("PipeOpTorchGroupNorm",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_group_norm", param_vals = list()) {
      param_set = ps(
        num_groups = p_int(lower = 1L, tags = c("train", "required")),
        eps = p_dbl(default = 1e-5, lower = 0, tags = "train"),
        affine = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_group_norm
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      # the number of dimensions is checked first, so that a shape that is too short is not
      # reported as having an unknown feature dimension
      assert_ndim(shape, min = 2L, id = self$id)
      assert_known_dims(shape, 2L, "the feature dimension (dimension 2)", self$id)
      num_groups = param_vals[["num_groups"]]
      if (shape[2L] %% num_groups != 0L) {
        stopf("PipeOp '%s' requires 'num_groups' (%i) to divide the number of channels of the input shape %s, which is %i.", # nolint
          self$id, num_groups, shape_to_str(shape), shape[2L])
      }
      shapes_in
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$num_channels = shapes_in[[1L]][2L]
      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_group_norm", PipeOpTorchGroupNorm)
