#' @title Dropout
#' @inherit torch::nnf_dropout description
#' @section nn_module:
#' Calls [`torch::nn_dropout()`] when trained.
#' @section Parameters:
#' * `p` :: `numeric(1)`\cr
#'  Probability of an element to be zeroed. Default: 0.5.
#' * `inplace` :: `logical(1)`\cr
#'   If set to `TRUE`, will do this operation in-place. Default: `FALSE`.
#'
#' @templateVar id nn_dropout
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchDropout = R6Class("PipeOpTorchDropout",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_dropout", param_vals = list()) {
      param_set = ps(
        p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_dropout
      )
    }
  )
)

# Base class for the channel-wise dropouts, which zero whole feature maps instead of single
# elements and therefore expect an input of a fixed rank.
PipeOpTorchDropoutNd = R6Class("PipeOpTorchDropoutNd",
  inherit = PipeOpTorch,
  public = list(
    # @description
    # Creates a new instance of this [R6][R6::R6Class] class.
    # @template params_pipelines
    # @template param_module_generator
    # @param d (`integer(1)`)\cr
    #   The number of spatial dimensions of the input.
    initialize = function(id, module_generator, d, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, coerce = TRUE)
      param_set = ps(
        p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = module_generator
      )
    }
  ),
  private = list(
    .d = NULL,
    .additional_phash_input = function() {
      list(private$.d)
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      # torch accepts an input with one dimension too few, but only warns that it is guessing which
      # dimension holds the channels; here the rank is known, so the input is rejected instead
      assert_ndim(shapes_in[[1L]], private$.d + 2L, self$id)
      shapes_in
    }
  )
)

#' @title 2D Dropout
#' @inherit torch::nnf_dropout2d description
#' @section nn_module:
#' Calls [`torch::nn_dropout2d()`] when trained.
#' @section Parameters:
#' * `p` :: `numeric(1)`\cr
#'  Probability of a channel to be zeroed. Default: 0.5.
#' * `inplace` :: `logical(1)`\cr
#'   If set to `TRUE`, will do this operation in-place. Default: `FALSE`.
#'
#' @templateVar id nn_dropout2d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchDropout2D = R6Class("PipeOpTorchDropout2D",
  inherit = PipeOpTorchDropoutNd,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_dropout2d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_dropout2d, d = 2, param_vals = param_vals)
    }
  )
)

#' @title 3D Dropout
#' @inherit torch::nnf_dropout3d description
#' @section nn_module:
#' Calls [`torch::nn_dropout3d()`] when trained.
#' @inheritSection mlr_pipeops_nn_dropout2d Parameters
#'
#' @templateVar id nn_dropout3d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchDropout3D = R6Class("PipeOpTorchDropout3D",
  inherit = PipeOpTorchDropoutNd,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_dropout3d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_dropout3d, d = 3, param_vals = param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_dropout", PipeOpTorchDropout)
register_po("nn_dropout2d", PipeOpTorchDropout2D)
register_po("nn_dropout3d", PipeOpTorchDropout3D)
