#' @title Average Pooling
#' @description
#' 1D, 2D and 3D Average Pooling.
#' @section Calls:
#' * `nn_avg_pool1d`
#' * `nn_avg_pool2d`
#' * `nn_avg_pool3d`
#'
#' @name avg_pool
NULL

#' @template param_id
#' @template param_param_vals
#' @rdname avg_pool
#' @export
TorchOpAvgPool1D = R6Class("TorchOpAvgPool1D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "avg_pool1d", param_vals = list()) {
      param_set = make_paramset_avg_pool(d = 1L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      assert_true(length(inputs$input$shape) == 3L)
      invoke(nn_avg_pool1d, .args = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname avg_pool
#' @export
TorchOpAvgPool2D = R6Class("TorchOpAvgPool1D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "avg_pool2d", param_vals = list()) {
      param_set = make_paramset_avg_pool(d = 2L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      assert_true(length(inputs$input$shape) == 4L)
      invoke(nn_avg_pool2d, .args = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname avg_pool
#' @export
TorchOpAvgPool3D = R6Class("TorchOpAvgPool1D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "avg_pool3d", param_vals = list()) {
      param_set = make_paramset_avg_pool(d = 3L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      assert_true(length(inputs$input$shape) == 5L)
      invoke(nn_avg_pool3d, .args = param_vals)
    }
  )
)



make_paramset_avg_pool = function(d) {
  force(d)
  check_fn = function(x) {
    if (is.null(x) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) {
      return(TRUE)
    }
    sprintf("Must be an integerish vector of length 1 or %s", d)
  }

  param_set = ps(
    kernel_size = p_uty(custom_check = check_fn, tags = c("required", "train")),
    stride = p_uty(default = NULL, custom_check = check_fn, tags = "train"),
    padding = p_uty(default = 0L, custom_check = check_fn, tags = "train"),
    ceil_mode = p_lgl(default = FALSE, tags = "train"),
    count_include_pad = p_lgl(default = TRUE, tags = "train")
  )
  if (d >= 2L) {
    param_set$add(ParamDbl$new("divisor_override", lower = 0.00001, tags = "train"))
  }

  param_set
}

#' @include mlr_torchops.R
mlr_torchops$add("avg_pool1d", TorchOpAvgPool1D)
#' @include mlr_torchops.R
mlr_torchops$add("avg_pool2d", TorchOpAvgPool2D)
#' @include mlr_torchops.R
mlr_torchops$add("avg_pool3d", TorchOpAvgPool3D)
