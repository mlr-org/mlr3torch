#' @rawNamespace exportPattern("^PipeOpPreprocTorch")
NULL

# The shape rules below describe what the wrapped `torchvision` functions actually do, which is
# not always what one would expect: `transform_crop()` clamps the crop to the image while
# `transform_center_crop()` pads it, and `transform_resize()` preserves the aspect ratio when
# `size` is a single number. They are therefore verified against the output of the wrapped
# functions in `tests/testthat/test_shape_inference.R`.
#
# The spatial dimensions are the last two and the channel dimension is the third from last, which
# holds both for the functions that are applied `rowwise` -- they see `(channels, height, width)`
# -- and for those that see the whole batch.

# replaces the last `length(values)` dimensions of `shape`
replace_tail = function(shape, values) {
  c(shape[seq_len(length(shape) - length(values))], as.integer(values))
}

# these operators are applied per row, so the shape needs a batch dimension in front of the image
assert_image_shape = function(shape, id, ndim = 4L) {
  if (length(shape) >= ndim) {
    return(invisible(shape))
  }
  stopf("PipeOp '%s' requires an input with at least %i dimensions (batch, channels, height, width), but got the shape %s, which has %i.", # nolint
    id, ndim, shape_to_str(shape), length(shape))
}

# Extent of `transform_crop()` along one axis. The crop is `img[.., start:(start + size - 1)]`, so
# it is clamped to the image and may be empty; it is never padded.
crop_extent = function(extent, start, size) {
  pmax(0L, pmin(as.integer(size), extent - as.integer(start) + 1L))
}

# Extent of `transform_resize()`. A `size` of length 2 is used as is, a single number matches the
# *shorter* side and scales the other one accordingly (truncating).
resize_extent = function(hw, size) {
  if (length(size) >= 2L) {
    return(as.integer(size[1:2]))
  }
  if (anyNA(hw) || any(hw == 0L)) {
    return(c(NA_integer_, NA_integer_))
  }
  scaled = as.integer(size * max(hw) / min(hw))
  if (hw[1L] <= hw[2L]) c(as.integer(size), scaled) else c(scaled, as.integer(size))
}

resize_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  # `trafo_resize` is applied to the whole batch, so it needs the batch dimension and an image
  assert_image_shape(shape, self$id, ndim = 3L)
  list(replace_tail(shape, resize_extent(utils::tail(shape, 2L), param_vals[["size"]])))
}

# `padding` is either applied to all sides, or is (left/right, top/bottom), or
# (left, right, top, bottom) -- note that torchvision's R port differs from PyTorch here
pad_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  # only the trailing two dimensions are touched, so a channel-less image is fine
  assert_image_shape(shape, self$id, ndim = 3L)
  padding = as.integer(param_vals[["padding"]])
  added = switch(as.character(length(padding)),
    "1" = c(2L * padding, 2L * padding),
    "2" = c(2L * padding[2L], 2L * padding[1L]),
    "4" = c(padding[3L] + padding[4L], padding[1L] + padding[2L]),
    stopf("PipeOp '%s' requires 'padding' to have 1, 2 or 4 values, but it has %i.", self$id, length(padding))
  )
  extent = utils::tail(shape, 2L) + added
  # negative padding crops and can make the image empty
  assert_positive_extent(extent, shape, self$id)
  list(replace_tail(shape, extent))
}

# `transform_rgb_to_grayscale()` and everything built on it index the first three channels
assert_rgb_channels = function(shape, id) {
  channels = shape[length(shape) - 2L]
  if (!is.na(channels) && channels < 3L) {
    stopf("PipeOp '%s' requires an RGB image, i.e. at least 3 channels, but the input shape %s has %i.", # nolint
      id, shape_to_str(shape), channels)
  }
  invisible(shape)
}

# an image with more than three channels is truncated to three
truncate_rgb_channels = function(shape) {
  channels = shape[length(shape) - 2L]
  shape[length(shape) - 2L] = if (!is.na(channels) && channels >= 3L) 3L else NA_integer_
  shape
}

grayscale_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  # `transform_grayscale()` repeats with a fixed length-4 specification, which only matches the
  # channel dimension when it is applied to a single (channels, height, width) image
  assert_ndim(shape, 4L, self$id)
  assert_rgb_channels(shape, self$id)
  shape[2L] = as.integer(param_vals[["num_output_channels"]])
  list(shape)
}

# `transform_rgb_to_grayscale()` *drops* the channel dimension instead of setting it to 1
rgb_to_grayscale_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  assert_image_shape(shape, self$id)
  assert_rgb_channels(shape, self$id)
  list(shape[-(length(shape) - 2L)])
}

crop_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  # only the trailing two dimensions are touched, so a channel-less image is fine
  assert_image_shape(shape, self$id, ndim = 3L)
  hw = utils::tail(shape, 2L)
  list(replace_tail(shape, c(
    crop_extent(hw[1L], param_vals[["top"]], param_vals[["height"]]),
    crop_extent(hw[2L], param_vals[["left"]], param_vals[["width"]])
  )))
}

# `transform_center_crop()` pads when the image is smaller than `size`, and torchvision swaps
# height and width while doing so, which makes the padded case unpredictable. The output extent is
# therefore only known where no padding is needed.
center_crop_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  # only the trailing two dimensions are touched, so a channel-less image is fine
  assert_image_shape(shape, self$id, ndim = 3L)
  size = as.integer(rep(param_vals[["size"]], length.out = 2L))
  assert_positive_extent(size, shape, self$id)
  hw = utils::tail(shape, 2L)
  # torchvision swaps height and width in the padded branch, which corrupts *both* extents, so
  # neither is known as soon as one of them needs padding
  if (any(is.na(hw) | hw < size)) {
    size = c(NA_integer_, NA_integer_)
  }
  list(replace_tail(shape, size))
}

# `transform_resized_crop()` crops and then resizes, so both rules have to be composed: with a
# single `size` the aspect ratio of the *cropped* image decides the output
resized_crop_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  # only the trailing two dimensions are touched, so a channel-less image is fine
  assert_image_shape(shape, self$id, ndim = 3L)
  hw = utils::tail(shape, 2L)
  cropped = c(
    crop_extent(hw[1L], param_vals[["top"]], param_vals[["height"]]),
    crop_extent(hw[2L], param_vals[["left"]], param_vals[["width"]])
  )
  # resizing an empty crop is impossible
  assert_positive_extent(cropped, shape, self$id)
  list(replace_tail(shape, resize_extent(cropped, param_vals[["size"]])))
}

# `transform_color_jitter()` delegates to `transform_adjust_hue()` when `hue` is set, which keeps
# only the first three channels
color_jitter_shapes = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  assert_image_shape(shape, self$id)
  hue = param_vals[["hue"]]
  if (is.null(hue) || all(hue == 0)) {
    return(shapes_in)
  }
  assert_rgb_channels(shape, self$id)
  list(truncate_rgb_channels(shape))
}

reshape_shapes = function(shapes_in, param_vals, task) {
  list(reshape_output_shape(shapes_in[[1L]], param_vals[["shape"]], self$id))
}

# operators that convert between colour spaces need three channels; an unknown channel count is
# accepted, because the network may still be valid at runtime
unchanged_shapes_rgb = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  assert_image_shape(shape, self$id)
  assert_rgb_channels(shape, self$id)
  # the hue is adjusted in HSV space, which keeps only the first three channels
  list(truncate_rgb_channels(shape))
}

unchanged_shapes_image = function(shapes_in, param_vals, task) {
  shape = shapes_in[[1L]]
  assert_image_shape(shape, self$id)
  assert_rgb_channels(shape, self$id)
  shapes_in
}

unchanged_shapes = function(shapes_in, param_vals, task) {
  shapes_in
}

#' @title Resizing Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_resize
register_preproc("trafo_resize", torchvision::transform_resize,
  packages = "torchvision",
  param_set = ps(
    size = p_uty(tags = c("train", "required"), custom_check = crate(function(x) {
      check_integerish(x, min.len = 1L, max.len = 2L, any.missing = FALSE)
    })),
    interpolation = p_fct(levels = c("Undefined", "Bartlett", "Blackman", "Bohman", "Box", "Catrom", "Cosine", "Cubic",
      "Gaussian", "Hamming", "Hann", "Hanning", "Hermite", "Jinc", "Kaiser", "Lagrange", "Lanczos", "Lanczos2",
      "Lanczos2Sharp", "LanczosRadius", "LanczosSharp", "Mitchell", "Parzen", "Point", "Quadratic", "Robidoux",
      "RobidouxSharp", "Sinc", "SincFast", "Spline", "Triangle", "Welch", "Welsh", "Bessel")
      , special_vals = list(0L, 2L, 3L),
      tags = "train", default = 2L
    )
  ),
  shapes_out = resize_shapes,
  rowwise = FALSE
)

#' @title No Transformation
#' @usage NULL
#' @name mlr_pipeops_trafo_nop
#' @aliases PipeOpPreprocTorchTrafoNop
#' @rdname PipeOpPreprocTorchTrafoNop
#' @format [`R6Class`][R6::R6Class] inheriting from [`PipeOpTaskPreprocTorch`].
#'
#' @description
#' Does nothing.
register_preproc("trafo_nop", identity, rowwise = FALSE, shapes_out = unchanged_shapes)

#' @title Reshaping Transformation
#' @usage NULL
#' @name mlr_pipeops_trafo_reshape
#' @aliases PipeOpPreprocTorchTrafoReshape
#' @rdname PipeOpPreprocTorchTrafoReshape
#' @format [`R6Class`][R6::R6Class] inheriting from [`PipeOpTaskPreprocTorch`].
#' @section Parameters:
#' * `shape` :: `integer()`\cr
#'   The desired output shape. The first dimension is the batch dimension and should usually be `-1`.
#'
#' @description
#' Reshapes the tensor according to the parameter `shape`, by calling `torch_reshape()`.
#' This preprocessing function is applied batch-wise.
register_preproc("trafo_reshape", torch_reshape, rowwise = FALSE, shapes_out = reshape_shapes,
  param_set = ps(
    shape = p_uty(tags = c("train", "required"), custom_check = check_integerish)
  )
)

#' @title Adjust Gamma Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_gamma
register_preproc("trafo_adjust_gamma", torchvision::transform_adjust_gamma, packages = "torchvision", shapes_out = unchanged_shapes, rowwise = TRUE,
  param_set = ps(
    gamma = p_dbl(lower = 0, tags = c("train", "required")),
    gain = p_dbl(default = 1, tags = "train")
  )
)

#' @title Adjust Brightness Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_brightness
register_preproc("trafo_adjust_brightness", torchvision::transform_adjust_brightness, packages = "torchvision",
  shapes_out = unchanged_shapes, rowwise = TRUE,
  param_set = ps(
    brightness_factor = p_dbl(lower = 0, tags = c("train", "required"))
  )
)

#' @title Adjust Hue Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_hue
register_preproc("trafo_adjust_hue", torchvision::transform_adjust_hue, packages = "torchvision",
  rowwise = TRUE, shapes_out = unchanged_shapes_rgb,
  param_set = ps(
    hue_factor = p_dbl(lower = -0.5, upper = 0.5, tags = c("train", "required"))
  )
)

#' @title Adjust Saturation Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_saturation
register_preproc("trafo_adjust_saturation", torchvision::transform_adjust_saturation, packages = "torchvision",
  # the saturation is adjusted via a grayscale image, which needs three channels
  shapes_out = unchanged_shapes_image, rowwise = TRUE,
  param_set = ps(
    saturation_factor = p_dbl(tags = c("train", "required"))
  )
)

#' @title Grayscale Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_grayscale
register_preproc("trafo_grayscale", torchvision::transform_grayscale, packages = "torchvision", shapes_out = grayscale_shapes, rowwise = TRUE,
  param_set = ps(
    num_output_channels = p_int(lower = 1L, upper = 3L, tags = c("train", "required"))
  )
)

#' @title RGB to Grayscale Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_rgb_to_grayscale
register_preproc("trafo_rgb_to_grayscale", torchvision::transform_rgb_to_grayscale, packages = "torchvision", shapes_out = rgb_to_grayscale_shapes, rowwise = TRUE,
  param_set = ps()
)

#' @title Normalization Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_normalize
register_preproc("trafo_normalize", torchvision::transform_normalize, packages = "torchvision", rowwise = TRUE,
  param_set = ps(
    mean = p_uty(tags = c("train", "required")),
    std = p_uty(tags = c("train", "required"))
    # no inplace parameter as this might be problematic when a preprocessing pipeop's output is connected to multiple
    # other pipeops
  ),
  shapes_out = function(shapes_in, param_vals, task) {
    s = shapes_in[[1L]]
    assert_true(length(s) >= 2, .var.name = "Input tensor must have at least 2 dimensions")
    shapes_in
  }
)

#' @title Padding Transformation
#' @template preprocess_torchvision
#' @templateVar id trafo_pad
register_preproc("trafo_pad", torchvision::transform_pad, packages = "torchvision", shapes_out = pad_shapes, rowwise = TRUE,
  param_set = ps(
    padding = p_uty(tags = c("train", "required"), custom_check = crate(function(x) check_integerish(x, min.len = 1L, max.len = 4L, any.missing = FALSE))),
    fill = p_uty(default = 0, tags = "train"),
    padding_mode = p_fct(default = "constant", levels = c("constant", "edge", "reflect", "symmetric"), tags = "train")
  )
)

# Data Augmentation

#' @title Resized Crop Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_resized_crop
register_preproc("augment_resized_crop", torchvision::transform_resized_crop, packages = "torchvision", shapes_out = resized_crop_shapes, rowwise = TRUE,
  param_set = ps(
    top = p_int(lower = 1L, tags = c("train", "required")),
    left = p_int(lower = 1L, tags = c("train", "required")),
    height = p_int(lower = 0L, tags = c("train", "required")),
    width = p_int(lower = 0L, tags = c("train", "required")),
    size = p_uty(tags = c("train", "required"), custom_check = crate(function(x) check_integerish(x, min.len = 1L, max.len = 2L, any.missing = FALSE))),
    interpolation = p_int(default = 2L, lower = 0L, upper = 3L, tags = "train")
  )
)

#' @title Color Jitter Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_color_jitter
register_preproc("augment_color_jitter", torchvision::transform_color_jitter, packages = "torchvision", shapes_out = color_jitter_shapes, rowwise = TRUE,
  param_set = ps(
    brightness = p_dbl(default = 0, lower = 0, tags = "train"),
    contrast = p_dbl(default = 0, lower = 0, tags = "train"),
    saturation = p_dbl(default = 0, lower = 0, tags = "train"),
    hue = p_dbl(default = 0, lower = 0, tags = "train")
  )
)

#' @title Random Resized Crop Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_resized_crop
register_preproc("augment_random_resized_crop", torchvision::transform_random_resized_crop, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    size = p_uty(tags = c("train", "required")),
    scale = p_uty(default = c(0.08, 1), tags = "train"),
    ratio = p_uty(default = c(3 / 4, 4 / 3), tags = "train"),
    interpolation = p_int(default = 2L, lower = 0L, upper = 3L, tags = "train")
  )
)

#' @title Random Order Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_order
register_preproc("augment_random_order", torchvision::transform_random_order, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    transforms = p_uty(tags = c("train", "required"), custom_check = crate(function(x) check_list(x, min.len = 1L)))
  )
)

#' @title Horizontal Flip Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_hflip
register_preproc("augment_hflip", torchvision::transform_hflip, packages = "torchvision", rowwise = TRUE,
  # a flip reads no dimension of the input, so nothing has to be known about it
  shapes_out = unchanged_shapes,
  param_set = ps()
)

#' @title Random Horizontal Flip Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_horizontal_flip
register_preproc("augment_random_horizontal_flip", torchvision::transform_random_horizontal_flip, packages = "torchvision", rowwise = TRUE,
  param_set = ps(
    p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
  ),
  # a flip reads no dimension of the input, so nothing has to be known about it
  shapes_out = unchanged_shapes
)

#' @title Crop Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_crop
register_preproc("augment_crop", torchvision::transform_crop, packages = "torchvision", shapes_out = crop_shapes, rowwise = TRUE,
  param_set = ps(
    top = p_int(lower = 1L, tags = c("train", "required")),
    left = p_int(lower = 1L, tags = c("train", "required")),
    height = p_int(lower = 0L, tags = c("train", "required")),
    width = p_int(lower = 0L, tags = c("train", "required"))
  )
)

#' @title Random Vertical Flip Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_vertical_flip
register_preproc("augment_random_vertical_flip", torchvision::transform_random_vertical_flip, packages = "torchvision",
  # a flip reads no dimension of the input, so nothing has to be known about it
  rowwise = TRUE, shapes_out = unchanged_shapes,
  param_set = ps(
    p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
  )
)

#' @title Random Affine Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_affine
register_preproc("augment_random_affine", torchvision::transform_random_affine, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    degrees = p_uty(tags = c("train", "required")),
    translate = p_uty(default = NULL, tags = "train"),
    scale = p_uty(default = NULL, tags = "train"),
    resample = p_int(default = 0, tags = "train"),
    fillcolor = p_uty(default = 0, tags = "train")
  )
)


#' @title Vertical Flip Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_vflip
register_preproc("augment_vflip", torchvision::transform_vflip, packages = "torchvision", rowwise = TRUE,
  param_set = ps(),
  # a flip reads no dimension of the input, so nothing has to be known about it
  shapes_out = unchanged_shapes
)

#' @title Rotate Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_rotate
register_preproc("augment_rotate", torchvision::transform_rotate, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    angle = p_uty(tags = c("train", "required")),
    resample = p_int(default = 0L, tags = "train"),
    expand = p_lgl(default = FALSE, tags = "train"),
    center = p_uty(default = NULL, tags = "train"),
    fill = p_uty(default = NULL, tags = "train")
  )
)

#' @title Center Crop Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_center_crop
register_preproc("augment_center_crop", torchvision::transform_center_crop, packages = "torchvision", shapes_out = center_crop_shapes, rowwise = TRUE,
  param_set = ps(
    size = p_uty(tags = c("train", "required"), custom_check = crate(function(x) check_integerish(x, lower = 1L, min.len = 1L, max.len = 2L, any.missing = FALSE))) # nolint
  )
)

#' @title Random Crop Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_crop
register_preproc("augment_random_crop", torchvision::transform_random_crop, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    size = p_uty(tags = c("train", "required")),
    padding = p_uty(default = NULL, tags = "train"),
    pad_if_needed = p_lgl(default = FALSE, tags = "train"),
    fill = p_uty(default = 0L, tags = "train"),
    padding_mode = p_fct(default = "constant", levels = c("constant", "edge", "reflect", "symmetric"), tags = "train")
  )
)

#' @title Random Choice Augmentation
#' @template preprocess_torchvision
#' @templateVar id augment_random_choice
register_preproc("augment_random_choice", torchvision::transform_random_choice, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    transforms = p_uty(tags = c("train", "required"), custom_check = crate(function(x) check_list(x, min.len = 1L)))
  )
)

##' @title Random Rotation Augmentation
##' @name PipeOpPreprocTorchAugmentRandomRotation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_rotation", torchvision::transform_random_rotation, packages = "torchvision", shapes_out = NULL, rowwise = TRUE)

##' @title Random Erasing Augmentation
##' @name PipeOpPreprocTorchAugmentRandomErasing
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_erasing", torchvision::transform_random_erasing, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

##' @title Perspective Augmentation
##' @name PipeOpPreprocTorchAugmentPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_perspective", torchvision::transform_perspective, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)


# not implemented for tensor
##' @title Random Grayscale Augmentation
##' @name PipeOpPreprocTorchAugmentRandomGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_grayscale", torchvision::transform_random_grayscale, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

# infering shape does not work, we could do it manually
##' @title Linear Transformation Augmentation
##' @name PipeOpPreprocTorchAugmentLinearTransformation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_linear_transformation", torchvision::transform_linear_transformation, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

##' @title Random Perspective Augmentation
##' @template preprocess_torchvision
##' @templateVar id augment_random_perspective
#register_preproc("augment_random_perspective", torchvision::transform_random_perspective, packages = "torchvision", shapes_out = NULL, rowwise = TRUE)

##' @title Random Apply Augmentation
##' @template preprocess_torchvision
##' @templateVar id augment_random_apply
#register_preproc("augment_random_apply", torchvision::transform_random_apply, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
#  param_set = ps(
#    transforms = p_uty(tags = c("train", "required"), custom_check = crate(function(x) check_list(x, min.len = 1L))),
#    p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
#  )
#)
