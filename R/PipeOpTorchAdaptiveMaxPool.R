PipeOpTorchAdaptiveMaxPool = R6Class("PipeOpTorchAdaptiveMaxPool",
  inherit = PipeOpTorch,
  public = list(
    #  @description Creates a new instance of this [R6][R6::R6Class] class.
    #  @template params_pipelines
    #  @param d (`integer(1)`)\cr
    #    The dimension of the adaptive max pooling operation.
    initialize = function(id, d, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3, coerce = TRUE)
      module_generator = switch(private$.d,
        nn_adaptive_max_pool1d, nn_adaptive_max_pool2d, nn_adaptive_max_pool3d)
      param_set = ps(
        output_size = p_uty(custom_check = make_check_vector(private$.d, null_ok = FALSE),
          tags = c("required", "train"))
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = module_generator,
        outname = c("output", "indices")
      )
      # the indices are only an output channel when they are requested
      self$outputs = "output"
    }
  ),
  private = list(
    .additional_phash_input = function() {
      c(super$.additional_phash_input(), list(private$.d))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      # a pooling operator over `d` dimensions expects `(batch, channels, <d spatial dimensions>)`.
      assert_ndim(shapes_in[[1L]], private$.d + 2L, self$id)
      res = list(adaptive_output_shape(
        shape_in = shapes_in[[1L]],
        conv_dim = private$.d,
        output_size = param_vals[["output_size"]],
        id = self$id
      ))

      # the indices have the same shape as the pooled output, they say where each maximum came from
      rep(res, 2)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      # the indices are only computed when they are among `$outputs`
      c(param_vals, list(return_indices = "indices" %in% self$output$name))
    },
    # torch returns the indices together with the output, or the output alone
    .module_outputs = function() {
      if ("indices" %in% self$output$name) c("output", "indices") else "output"
    },
    .d = NULL
  )
)

#' @title 1D Adaptive Max Pooling
#'
#' @inherit torch::nnf_adaptive_max_pool1d description
#' @section nn_module:
#' Calls [`nn_adaptive_max_pool1d()`][torch::nn_adaptive_max_pool1d] during training.
#' @section Parameters:
#' * `output_size` :: `integer(1)`\cr
#'   The target output size. A single number.
#' @templateVar id nn_adaptive_max_pool1d
#' @section Input and Output Channels:
#' There is one input channel `"input"`.
#' The module has two outputs, `"output"` and `"indices"`, of which only `"output"` is an output
#' channel by default. Set `$outputs` to also (or only) get the indices, e.g.
#' `outputs = c("output", "indices")`, which are only computed when they are among `$outputs`.
#' For an explanation see [`PipeOpTorch`].
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchAdaptiveMaxPool1D = R6Class("PipeOpTorchAdaptiveMaxPool1D", inherit = PipeOpTorchAdaptiveMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_max_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, param_vals = param_vals)
    }
  )
)

#' @title 2D Adaptive Max Pooling
#'
#' @inherit torch::nnf_adaptive_max_pool2d description
#' @section nn_module:
#' Calls [`nn_adaptive_max_pool2d()`][torch::nn_adaptive_max_pool2d] during training.
#' @section Parameters:
#' * `output_size` :: `integer()`\cr
#'   The target output size. Can be a single number or a vector.
#' @templateVar id nn_adaptive_max_pool2d
#' @inheritSection mlr_pipeops_nn_adaptive_max_pool1d Input and Output Channels
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchAdaptiveMaxPool2D = R6Class("PipeOpTorchAdaptiveMaxPool2D", inherit = PipeOpTorchAdaptiveMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_max_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, param_vals = param_vals)
    }
  )
)

#' @title 3D Adaptive Max Pooling
#'
#' @inherit torch::nnf_adaptive_max_pool3d description
#' @section nn_module:
#' Calls [`nn_adaptive_max_pool3d()`][torch::nn_adaptive_max_pool3d] during training.
#' @inheritSection mlr_pipeops_nn_adaptive_max_pool2d Parameters
#' @templateVar id nn_adaptive_max_pool3d
#' @inheritSection mlr_pipeops_nn_adaptive_max_pool1d Input and Output Channels
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchAdaptiveMaxPool3D = R6Class("PipeOpTorchAdaptiveMaxPool3D", inherit = PipeOpTorchAdaptiveMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_max_pool3d", param_vals = list()) {
      super$initialize(id = id, d = 3, param_vals = param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_adaptive_max_pool1d", PipeOpTorchAdaptiveMaxPool1D)
register_po("nn_adaptive_max_pool2d", PipeOpTorchAdaptiveMaxPool2D)
register_po("nn_adaptive_max_pool3d", PipeOpTorchAdaptiveMaxPool3D)
