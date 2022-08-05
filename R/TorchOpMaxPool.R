#' @title Max Pooling
#' @description
#' 1D, 2D and 3D Max Pooling.
#' @section Calls:
#' * `nn_max_pool1d`
#' * `nn_max_pool2d`
#' * `nn_max_pool3d`
#'
#' @name max_pool
NULL

#' @param return_indices (`logical(1)`)\cr
#'   Whether to return the inidices.
#' @template param_id
#' @template param_param_vals
#' @rdname max_pool
#' @export
TorchOpMaxPool1D = R6Class("TorchOpMaxPool1D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(return_indices = FALSE, id = "max_pool1d", param_vals = list()) {
      param_set = make_paramset_max_pool(d = 1L)
      private$.return_indices = assert_flag(return_indices)
      if (return_indices) {
        output = data.table(
          name = c("output", "indices"),
          train = c("ModelConfig", "ModelConfig"),
          predict = c("ModelConfig", "ModelConfig")
        )
      } else {
        output = data.table(name = "output", train = "ModelConfig", predict = "ModelConfig")
      }
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        output = output
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      assert_true(length(inputs$input$shape) == 3L)
      invoke(nn_max_pool1d, .args = param_vals, return_indices = private$.return_indices)
    },
    .return_indices = NULL
  )
)

#' @param return_indices (`logical(1)`)\cr
#'   Whether to return the inidices.
#' @template param_id
#' @template param_param_vals
#' @rdname max_pool
#' @export
TorchOpMaxPool2D = R6Class("TorchOpMaxPool2D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(return_indices = FALSE, id = "max_pool2d", param_vals = list()) {
      param_set = make_paramset_max_pool(d = 2L)
      private$.return_indices = assert_flag(return_indices)
      if (return_indices) {
        output = data.table(
          name = c("output", "indices"),
          train = c("ModelConfig", "ModelConfig"),
          predict = c("ModelConfig", "ModelConfig")
        )
      } else {
        output = data.table(name = "output", train = "ModelConfig", predict = "ModelConfig")
      }
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        output = output
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      assert_true(length(inputs$input$shape) == 4L)
      invoke(nn_max_pool2d, .args = param_vals, return_indices = private$.return_indices)
    },
    .return_indices = NULL
  )
)

#' @param return_indices (`logical(1)`)\cr
#'   Whether to return the inidices.
#' @template param_id
#' @template param_param_vals
#' @rdname max_pool
#' @export
TorchOpMaxPool3D = R6Class("TorchOpMaxPool3D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(return_indices = FALSE, id = "max_pool3d", param_vals = list()) {
      param_set = make_paramset_max_pool(d = 3L)
      private$.return_indices = assert_flag(return_indices)
      if (return_indices) {
        output = data.table(
          name = c("output", "indices"),
          train = c("ModelConfig", "ModelConfig"),
          predict = c("ModelConfig", "ModelConfig")
        )
      } else {
        output = data.table(name = "output", train = "ModelConfig", predict = "ModelConfig")
      }
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        output = output
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      assert_true(length(inputs$input$shape) == 5L)
      invoke(nn_max_pool3d, .args = param_vals, return_indices = private$.return_indices)
    },
    .return_indices = NULL
  )
)



make_paramset_max_pool = function(d) {
  force(d)
  check_fn = function(x) {
    if (is.null(x) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) {
      return(TRUE)
    }
    sprintf("Must be an integerish vector of length 1 or %s", d)
  }

  ps(
    kernel_size = p_uty(custom_check = check_fn, tags = c("required", "train")),
    stride = p_uty(default = NULL, custom_check = check_fn, tags = "train"),
    padding = p_uty(default = 0L, custom_check = check_fn, tags = "train"),
    dilation = p_int(default = 1L, tags = "train"),
    ceil_mode = p_lgl(default = FALSE, tags = "train")
  )
}

#' @include mlr_torchops.R
mlr_torchops$add("max_pool1d", TorchOpMaxPool1D)
#' @include mlr_torchops.R
mlr_torchops$add("max_pool2d", TorchOpMaxPool2D)
#' @include mlr_torchops.R
mlr_torchops$add("max_pool3d", TorchOpMaxPool3D)
