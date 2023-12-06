#' @rawNamespace exportPattern("^PipeOpPreprocTorch")
NULL

# FIXME: Better docs, not one for all

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

#' @title PipeOpPreprocTorchTrafoNop
#' @usage NULL
#' @name mlr_pipeops_preproc_torch.trafo_nop
#' @rdname PipeOpPreprocTorchTrafoNop
#' @format [`R6Class`] inheriting from [`PipeOpTaskPreprocTorch`].
#'
#' @description
#' Does nothing.
register_preproc("trafo_nop", identity, shapes_out = "unchanged", rowwise = FALSE)

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_gamma
register_preproc("trafo_adjust_gamma", torchvision::transform_adjust_gamma, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_brightness
register_preproc("trafo_adjust_brightness", torchvision::transform_adjust_brightness, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_hue
register_preproc("trafo_adjust_hue", torchvision::transform_adjust_hue, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE)

#' @template preprocess_torchvision
#' @templateVar id trafo_adjust_saturation
register_preproc("trafo_adjust_saturation", torchvision::transform_adjust_saturation, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id trafo_grayscale
register_preproc("trafo_grayscale", torchvision::transform_grayscale, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id trafo_rgb_to_grayscale
register_preproc("trafo_rgb_to_grayscale", torchvision::transform_rgb_to_grayscale, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id trafo_normalize
register_preproc("trafo_normalize", torchvision::transform_normalize, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id trafo_pad
register_preproc("trafo_pad", torchvision::transform_pad, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

# Data Augmentation

#' @template preprocess_torchvision
#' @templateVar id augment_resized_crop
register_preproc("augment_resized_crop", torchvision::transform_resized_crop, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_color_jitter
register_preproc("augment_color_jitter", torchvision::transform_color_jitter, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_resized_crop
register_preproc("augment_random_resized_crop", torchvision::transform_random_resized_crop, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_order
register_preproc("augment_random_order", torchvision::transform_random_order, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_hflip
register_preproc("augment_hflip", torchvision::transform_hflip, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_horizontal_flip
register_preproc("augment_random_horizontal_flip", torchvision::transform_random_horizontal_flip, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint


#' @template preprocess_torchvision
#' @templateVar id augment_crop
register_preproc("augment_crop", torchvision::transform_crop, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_vertical_flip
register_preproc("augment_random_vertical_flip", torchvision::transform_random_vertical_flip, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_affine
register_preproc("augment_random_affine", torchvision::transform_random_affine, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint


#' @template preprocess_torchvision
#' @templateVar id augment_vflip
register_preproc("augment_vflip", torchvision::transform_vflip, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_apply
register_preproc("augment_random_apply", torchvision::transform_random_apply, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_rotate
register_preproc("augment_rotate", torchvision::transform_rotate, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_center_crop
register_preproc("augment_center_crop", torchvision::transform_center_crop, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_choice
register_preproc("augment_random_choice", torchvision::transform_random_choice, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint

#' @template preprocess_torchvision
#' @templateVar id augment_random_crop
register_preproc("augment_random_crop", torchvision::transform_random_crop, packages = "torchvision", shapes_out = NULL, rowwise = TRUE)


##' @name PipeOpPreprocTorchAugmentRandomRotation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_rotation", torchvision::transform_random_rotation, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint

##' @name PipeOpPreprocTorchAugmentRandomErasing
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_erasing", torchvision::transform_random_erasing, packages = "torchvision", shapes_out = "infer", rowwise = TRUE)

# not implemented
##' @name PipeOpPreprocTorchAugmentPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_perspective", torchvision::transform_perspective, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

# not implemented for tensor
##' @name PipeOpPreprocTorchAugmentRandomGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_grayscale", torchvision::transform_random_grayscale, packages = "torchvision", shapes_out = "unchanged", rowwise = TRUE) # nolint

# infering shape does not work, we could do it manually
##' @name PipeOpPreprocTorchAugmentLinearTransformation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_linear_transformation", torchvision::transform_linear_transformation, packages = "torchvision", shapes_out = "infer", rowwise = TRUE) # nolint

##' @template preprocess_torchvision
##' @templateVar id augment_random_perspective
#register_preproc("augment_random_perspective", torchvision::transform_random_perspective, packages = "torchvision", shapes_out = NULL, rowwise = TRUE) # nolint