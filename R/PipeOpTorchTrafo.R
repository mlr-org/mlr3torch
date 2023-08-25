#' @title Resize a Tensor
#' @description
#' Resizes a tensor
#'
#' @section Parameters:
#' * `size` :: `integer()`\cr
#'   The size of the desired output image.
#' @export
PipeOpTorchImageResize = R6Class("PipeOpImageResize",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "image_resize",  param_vals = list()) {
      param_set = ps(
        size = p_uty(tags = c("train", "required"))
      )
      super$initialize(
        param_set = param_set,
        param_vals = param_vals,
        id = id,
        packages = "torchvision"
      )
    }
  ),
  private = list(
    .shapes_out  = function(shapes_in, param_vals, task) {
      list(c(NA_integer_, param_vals$size))
    },
    .make_module = function(shapes_in, param_vals, task) {
      crate(function(x) {
        torchvision::transform_resize(size = size)
      }, size = param_vals$size)
    }
  )
)



PipeOp
