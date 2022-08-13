#' @title Reshape Operations
#' @description
#' Reshapes a tensor to the given shape or squeezes / unsqueezes a tensor for the given dim
#' @name reshape_ops
NULL

#' @template param_id
#' @template param_param_vals
#' @rdname reshape_ops
#' @export
PipeOpTorchReshape = R6Class("PipeOpTorchReshape",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_reshape", param_vals = list()) {
      param_set = ps(
        shape = p_uty(tags = c("train", "required"), custom_check = check_integerish)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_reshape
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      shape = param_vals$shape
      shape[shape == -1] = NA
      assert_integerish(shape, lower = 1)
      if (sum(is.na(shape)) > 1) stop("'shape' must only contain one -1 or NA.")
      inlen = prod(shapes_in[[1]])
      outlen = prod(shape)
      # the following is going to trigger rarely, since the 1st dimension is typically NA
      if (!is.na(inlen) && !is.na(outlen) && inlen != outlen) stop("'shape' not compatible with input shape")
      list(shape)
    },
    .shape_dependent_params = function(shapes_in, param_vals) {
      param_vals$shape[is.na(param_vals$shape)] = -1
      param_vals
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname reshape_ops
#' @export
PipeOpTorchSqueeze = R6Class("PipeOpTorchSqueeze",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_squeeze", param_vals = list()) {
      param_set = ps(dim = p_int(tags = c("train", "required")))
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_squeeze
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      shape = shapes_in[[1]]
      true_dim = param_vals$dim
      if (true_dim < 0) {
        true_dim = 1 + length(shape) - true_dim
      }
      assert_int(true_dim, lower = 1, upper = length(shape))

      if (is.na(shape[[true_dim]])) stop("input shape for 'dim' dimension must be known.")
      if (shape[[true_dim]] == 1) shape = shape[-true_dim]

      list(shape)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname reshape_ops
#' @export
PipeOpTorchUnsqueeze = R6Class("PipeOpTorchUnqueeze",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_unsqueeze", param_vals = list()) {
      param_set = ps(dim = p_int(tags = c("train", "required")))
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_unsqueeze
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      shape = shapes_in[[1]]
      true_dim = param_vals$dim
      if (true_dim < 0) {
        true_dim = 1 + length(shape) - true_dim
      }
      assert_int(true_dim, lower = 1, upper = length(shape) + 1)
      list(append(shape, 1, after = true_dim - 1))
    }
  )
)

nn_reshape = nn_module(
  initialize = function(shape) {
    self$shape = shape
  },
  forward = function(input) {
    input$reshape(self$shape)
  }
)

nn_squeeze = nn_module(
  initialize = function(dim) {
    self$dim = dim
  },
  forward = function(input) {
    input$squeeze(self$dim)
  }
)

nn_unsqueeze = nn_module(
  initialize = function(dim) {
    self$dim = dim
  },
  forward = function(input) {
    input$unsqueeze(self$dim)
  }
)

#' @include mlr_torchops.R
register_po("nn_reshape", PipeOpTorchReshape)
register_po("nn_unsqueeze", PipeOpTorchUnsqueeze)
register_po("nn_squeeze", PipeOpTorchSqueeze)
