# The ranks that each interpolation mode is defined for, including the batch and channel dimension.
# `nearest` works on any number of spatial dimensions, the others are tied to one.
upsample_ranks = list(nearest = 3:5, linear = 3L, bilinear = 4L, bicubic = 4L, trilinear = 5L)

# The spatial extents that `nn_upsample()` produces. Exactly one of `size` and `scale_factor` is
# set; `size` is the extent itself, `scale_factor` multiplies the input extent and rounds down.
# @param spatial_in (`integer()`) The spatial extents of the input, possibly `NA`.
# @param size (`integer()` | `NULL`) The `size` parameter, of length 1 or `length(spatial_in)`.
# @param scale_factor (`numeric()` | `NULL`) The `scale_factor` parameter, same lengths.
# @param id (`character(1)`) The PipeOp's id, for the error messages.
upsample_output_extent = function(spatial_in, size, scale_factor, id) {
  d = length(spatial_in)
  if (is.null(size) == is.null(scale_factor)) {
    stopf("PipeOp '%s' requires exactly one of 'size' and 'scale_factor' to be set.", id)
  }
  given = size %??% scale_factor
  what = if (is.null(size)) "scale_factor" else "size"
  if (length(given) %nin% c(1L, d)) {
    stopf("PipeOp '%s' requires '%s' to have 1 or %i element(s), the number of spatial dimensions of the input, but it has %i.", # nolint
      id, what, d, length(given))
  }
  if (length(given) == 1L) given = rep(given, d)
  out = if (is.null(size)) floor(spatial_in * given) else given
  # `scale_factor` is a ratio, so a small one can round an extent down to zero
  if (any(!is.na(out) & out < 1)) {
    stopf("PipeOp '%s' cannot be applied to the spatial extents %s: '%s' would produce an output of size %s, which no tensor can have.", # nolint
      id, paste0(spatial_in, collapse = ", "), what, paste0(out, collapse = ", "))
  }
  as.integer(out)
}

#' @title Upsampling
#' @inherit torch::nn_upsample description
#' @section nn_module:
#' Calls [`torch::nn_upsample()`] when trained.
#' @section Parameters:
#' * `size` :: `integer()`\cr
#'   The spatial extents of the output, either a single number or one per spatial dimension.
#'   Exactly one of `size` and `scale_factor` has to be set.
#' * `scale_factor` :: `numeric()`\cr
#'   The factor by which the spatial extents of the input are multiplied, either a single number or
#'   one per spatial dimension. The resulting extents are rounded down.
#'   Exactly one of `size` and `scale_factor` has to be set.
#' * `mode` :: `character(1)`\cr
#'   The interpolation method, one of `"nearest"`, `"linear"`, `"bilinear"`, `"bicubic"` or
#'   `"trilinear"`. All but `"nearest"` are defined for one number of spatial dimensions only:
#'   `"linear"` for one, `"bilinear"` and `"bicubic"` for two and `"trilinear"` for three.
#'   Default is `"nearest"`.
#' * `align_corners` :: `logical(1)`\cr
#'   Whether the corner pixels of input and output are aligned, which only has an effect for the
#'   interpolating modes. Default is `NULL`, which torch treats as `FALSE`.
#'
#' @templateVar id nn_upsample
#' @templateVar param_vals scale_factor = 2
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchUpsample = R6Class("PipeOpTorchUpsample",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_upsample", param_vals = list()) {
      param_set = ps(
        size = p_uty(default = NULL, tags = "train", custom_check = crate(function(x) {
          check_integerish(x, lower = 1L, min.len = 1L, any.missing = FALSE, null.ok = TRUE)
        })),
        scale_factor = p_uty(default = NULL, tags = "train", custom_check = crate(function(x) {
          check_numeric(x, lower = 0, min.len = 1L, any.missing = FALSE, null.ok = TRUE)
        })),
        mode = p_fct(default = "nearest", levels = names(upsample_ranks), tags = "train"),
        align_corners = p_lgl(default = NULL, special_vals = list(NULL), tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_upsample
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      mode = param_vals[["mode"]] %??% "nearest"
      assert_ndim(shape, upsample_ranks[[mode]], self$id)
      spatial = upsample_output_extent(shape[-(1:2)], param_vals[["size"]],
        param_vals[["scale_factor"]], self$id)
      list(as.integer(c(shape[1:2], spatial)))
    }
  )
)

#' @include aaa.R
register_po("nn_upsample", PipeOpTorchUpsample)
