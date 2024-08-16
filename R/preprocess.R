#' @rawNamespace exportPattern("^PipeOpPreprocTorch")
NULL

#' @template preprocess_torchvision
#' @templateVar id trafo_resize
register_preproc("trafo_resize", torchvision::transform_resize,
  packages = "torchvision",
  param_set = ps(
    size = p_uty(tags = c("train", "required")),
    interpolation = p_fct(levels = c("Undefined", "Bartlett", "Blackman", "Bohman", "Box", "Catrom", "Cosine", "Cubic",
      "Gaussian", "Hamming", "Hann", "Hanning", "Hermite", "Jinc", "Kaiser", "Lagrange", "Lanczos", "Lanczos2",
      "Lanczos2Sharp", "LanczosRadius", "LanczosSharp", "Mitchell", "Parzen", "Point", "Quadratic", "Robidoux",
      "RobidouxSharp", "Sinc", "SincFast", "Spline", "Triangle", "Welch", "Welsh", "Bessel")
      , special_vals = list(0L, 2L, 3L),
      tags = "train", default = 2L
    )
  ),
  shapes_out = function(shapes_in, param_vals, task) {
    size = param_vals$size
    shape = shapes_in[[1L]]
    assert_true(length(shape) > 2)
    height = shape[[length(shape) - 1L]]
    width = shape[[length(shape)]]
    s = torchvision::transform_resize(torch_ones(c(1, height, width), device = "meta"), size = size)$shape[2:3]
    list(c(shape[seq_len(length(shape) - 2L)], s))
  },
  rowwise = FALSE
)

unchanged_shapes_rgb = function(shapes_in, param_vals, task) {
  assert_rgb_shape(shapes_in[[1L]])
  shapes_in
}

unchanged_shapes_image = function(shapes_in, param_vals, task) {
  assert_grayscale_or_rgb(shapes_in[[1L]])
  shapes_in
}

unchanged_shapes = function(shapes_in, param_vals, task) {
  shapes_in
}

#' @title PipeOpPreprocTorchTrafoNop
#' @usage NULL
#' @name mlr_pipeops_trafo_nop
#' @aliases PipeOpPreprocTorchTrafoNop
#' @rdname PipeOpPreprocTorchTrafoNop
#' @format [`R6Class`][R6::R6Class] inheriting from [`PipeOpTaskPreprocTorch`].
#'
#' @description
#' Does nothing.
register_preproc("trafo_nop", identity, rowwise = FALSE, shapes_out = unchanged_shapes)

#' @title PipeOpPreprocTorchTrafoReshape
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
register_preproc("trafo_reshape", torch_reshape, rowwise = FALSE, shapes_out = "infer",
  param_set = ps(
    shape = p_uty(tags = c("train", "required"), custom_check = check_integerish)
  )
)

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_gamma
register_preproc("trafo_adjust_gamma", torchvision::transform_adjust_gamma, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    gamma = p_dbl(lower = 0, tags = c("train", "required")),
    gain = p_dbl(default = 1, tags = "train")
  )
)

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_brightness
register_preproc("trafo_adjust_brightness", torchvision::transform_adjust_brightness, packages = "torchvision",
  shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    brightness_factor = p_dbl(lower = 0, tags = c("train", "required"))
  )
)

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_hue
register_preproc("trafo_adjust_hue", torchvision::transform_adjust_hue, packages = "torchvision",
  rowwise = TRUE, shapes_out = unchanged_shapes_rgb,
  param_set = ps(
    hue_factor = p_dbl(lower = -0.5, upper = 0.5, tags = c("train", "required"))
  )
)

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_saturation
register_preproc("trafo_adjust_saturation", torchvision::transform_adjust_saturation, packages = "torchvision",
  shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    saturation_factor = p_dbl(tags = c("train", "required"))
  )
)

#' @template preprocess_torchvision
#' @templateVar id trafo_grayscale
register_preproc("trafo_grayscale", torchvision::transform_grayscale, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    num_output_channels = p_int(lower = 1L, upper = 3L, tags = c("train", "required"))
  )
)

#' @template preprocess_torchvision
#' @templateVar id trafo_rgb_to_grayscale
register_preproc("trafo_rgb_to_grayscale", torchvision::transform_rgb_to_grayscale, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps()
)

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
    assert_true(length(s) >= 2)
    shapes_in
  }
)

#' @template preprocess_torchvision
#' @templateVar id trafo_pad
register_preproc("trafo_pad", torchvision::transform_pad, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    padding = p_uty(tags = c("train", "required")),
    fill = p_uty(default = 0, tags = "train"),
    padding_mode = p_fct(default = "constant", levels = c("constant", "edge", "reflect", "symmetric"), tags = "train")
  )
)

# Data Augmentation

#' @template preprocess_torchvision
#' @templateVar id augment_resized_crop
register_preproc("augment_resized_crop", torchvision::transform_resized_crop, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    top = p_int(tags = c("train", "required")),
    left = p_int(tags = c("train", "required")),
    height = p_int(tags = c("train", "required")),
    width = p_int(tags = c("train", "required")),
    size = p_uty(tags = c("train", "required")),
    interpolation = p_int(default = 2L, lower = 0L, upper = 3L, tags = "train")
  )
)

#' @template preprocess_torchvision
#' @templateVar id augment_color_jitter
register_preproc("augment_color_jitter", torchvision::transform_color_jitter, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    brightness = p_dbl(default = 0, lower = 0, tags = "train"),
    contrast = p_dbl(default = 0, lower = 0, tags = "train"),
    saturation = p_dbl(default = 0, lower = 0, tags = "train"),
    hue = p_dbl(default = 0, lower = 0, tags = "train")
  )
)

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

#' @template preprocess_torchvision
#' @templateVar id augment_random_order
register_preproc("augment_random_order", torchvision::transform_random_order, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    transforms = p_uty(tags = c("train", "required"), custom_check = check_list)
  )
)

#' @template preprocess_torchvision
#' @templateVar id augment_hflip
register_preproc("augment_hflip", torchvision::transform_hflip, packages = "torchvision", rowwise = TRUE,
  shapes_out = function(shapes_in, param_vals, task) {
    assert_grayscale_or_rgb(shapes_in[[1L]])
    shapes_in
  },
  param_set = ps()
)

#' @template preprocess_torchvision
#' @templateVar id augment_random_horizontal_flip
register_preproc("augment_random_horizontal_flip", torchvision::transform_random_horizontal_flip, packages = "torchvision", rowwise = TRUE,
  param_set = ps(
    p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
  ),
  shapes_out = function(shapes_in, param_vals, task) {
    assert_rgb_shape(shapes_in[[1L]])
    shapes_in
  }
)


#' @template preprocess_torchvision
#' @templateVar id augment_crop
register_preproc("augment_crop", torchvision::transform_crop, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    top = p_int(tags = c("train", "required")),
    left = p_int(tags = c("train", "required")),
    height = p_int(tags = c("train", "required")),
    width = p_int(tags = c("train", "required"))
  )
)

#' @template preprocess_torchvision
#' @templateVar id augment_random_vertical_flip
register_preproc("augment_random_vertical_flip", torchvision::transform_random_vertical_flip, packages = "torchvision",
  rowwise = TRUE, shapes_out = unchanged_shapes_rgb,
  param_set = ps(
    p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
  )
)

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


#' @template preprocess_torchvision
#' @templateVar id augment_vflip
register_preproc("augment_vflip", torchvision::transform_vflip, packages = "torchvision", rowwise = TRUE,
  param_set = ps(),
  shapes_out = function(shapes_in, param_vals, task) {
    assert_grayscale_or_rgb(shapes_in[[1L]])
    shapes_in
  }
)

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

#' @template preprocess_torchvision
#' @templateVar id augment_center_crop
register_preproc("augment_center_crop", torchvision::transform_center_crop, packages = "torchvision", shapes_out = "infer", rowwise = TRUE,
  param_set = ps(
    size = p_uty(tags = c("train", "required"))
  )
)

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

#' @template preprocess_torchvision
#' @templateVar id augment_random_choice
register_preproc("augment_random_choice", torchvision::transform_random_choice, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
  param_set = ps(
    transforms = p_uty(custom_check = check_list, tags = c("train", "required"))
  )
)


##' @name PipeOpPreprocTorchAugmentRandomRotation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_rotation", torchvision::transform_random_rotation, packages = "torchvision", shapes_out = NULL, rowwise = TRUE)

##' @name PipeOpPreprocTorchAugmentRandomErasing
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_erasing", torchvision::transform_random_erasing, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

# not implemented
##' @name PipeOpPreprocTorchAugmentPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_perspective", torchvision::transform_perspective, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

# not implemented for tensor
##' @name PipeOpPreprocTorchAugmentRandomGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_grayscale", torchvision::transform_random_grayscale, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

# infering shape does not work, we could do it manually
##' @name PipeOpPreprocTorchAugmentLinearTransformation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_linear_transformation", torchvision::transform_linear_transformation, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

##' @template preprocess_torchvision
##' @templateVar id augment_random_perspective
#register_preproc("augment_random_perspective", torchvision::transform_random_perspective, packages = "torchvision", shapes_out = NULL, rowwise = TRUE)

##' @template preprocess_torchvision
##' @templateVar id augment_random_apply
#register_preproc("augment_random_apply", torchvision::transform_random_apply, packages = "torchvision", shapes_out = NULL, rowwise = TRUE,
#  param_set = ps(
#    transforms = p_uty(tags = c("train", "required"), custom_check = check_list),
#    p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train")
#  )
#)

