#' @title Lazy Preprocessing and Transformations
#' @description
#' Overview over all implemented preprocessing methods.
#' See [`PipeOpTaskPreprocTorch`] for more details on the class itself.
#'
#' `PipeOps` preproessed with the `"augment_"` prefix, have their stages parameter initialize to `"train"`, while
#' those with the `"trafo_"` prefix have it set to `c("train", "predict")`.
#'
#' @usage NULL
#' @format NULL
#' @name mlr_pipeops_preproc_torch_overview
#' @rawNamespace exportPattern("^PipeOpPreprocTorch")
#' @include PipeOpTaskPreprocTorch.R
NULL

##' @name PipeOpPreprocTorchTrafoResize
##' @rdname mlr_pipeops_preproc_torch_overview
##' @description
##' * `trafo_resize`: Calls [`torchvision::transform_resize`]
#register_preproc("trafo_resize", torchvision::transform_resize,
#  packages = "torchvision",
#  param_set = ps(
#    size = p_uty(tags = c("train", "required")),
#    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
#      tags = "train", default = 2L
#    )
#  ),
#  shapes_out = function(shapes_in, param_vals, task) {
#    size = param_vals$size
#    shape = shapes_in[[1L]]
#    assert_true(length(shape) > 2)
#    height = shape[[length(shape) - 1L]]
#    width = shape[[length(shape)]]
#    s = torchvision::transform_resize(torch_ones(c(1, height, width), device = "meta"), size = size)$shape[2:3]
#    list(c(shape[seq_len(length(shape) - 2L)], s))
#  }
#)
#
##' @name PipeOpPreprocTorchTrafoNop
##' @rdname mlr_pipeops_preproc_torch_overview
##' @description
##' * `trafo_nop`: Calls `identity()`.
#register_preproc("trafo_nop", identity, shapes_out = NULL)
#
##' @name PipeOpPreprocTorchAdjustGamma
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("trafo_adjust_gamma", torchvision::transform_adjust_gamma, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchAdjustBrightness
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("trafo_adjust_brightness", torchvision::transform_adjust_brightness, packages = "torchvision") # nolint
#
##' @name PipeOpPreprocTorchAdjustHue
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("trafo_adjust_hue", torchvision::transform_adjust_hue, packages = "torchvision", shapes_out = "infer")
#
#
##' @name PipeOpPreprocTorchRandomCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_crop", torchvision::transform_random_crop, packages = "torchvision", shapes_out = NULL)
#
##' @name PipeOpPreprocTorchAdjustSaturation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("trafo_adjust_saturation", torchvision::transform_adjust_saturation, packages = "torchvision", shapes_out = "infer") # nolint
#
##' @name PipeOpPreprocTorchGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("grayscale", torchvision::transform_grayscale, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchRgbToGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("rgb_to_grayscale", torchvision::transform_rgb_to_grayscale, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchNormalize
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("trafo_normalize", torchvision::transform_normalize, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchPad
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("trafo_pad", torchvision::transform_pad, packages = "torchvision", shapes_out = "infer") # nolint
#
#
## Data Augmentation
#
##' @name PipeOpPreprocTorchResizedCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_resized_crop", torchvision::transform_resized_crop, packages = "torchvision", shapes_out = "infer") # nolint
#
#
##' @name PipeOpPreprocTorchColorJitter
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_color_jitter", torchvision::transform_color_jitter, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchRandomResizedCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_resized_crop", torchvision::transform_random_resized_crop, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchRandomOrder
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_order", torchvision::transform_random_order, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchHflip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_hflip", torchvision::transform_hflip, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchRandomRotation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_rotation", torchvision::transform_random_rotation, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchRandomHorizontalFlip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_horizontal_flip", torchvision::transform_random_horizontal_flip, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchLinaerTransformation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_linear_transformation", torchvision::transform_linear_transformation, packages = "torchvision", shapes_out = "infer") # nolint
#
##' @name PipeOpPreprocTorchCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_crop", torchvision::transform_crop, packages = "torchvision", shapes_out = "infer") # nolint
#
##' @name PipeOpPreprocTorchRandomVerticalFlip
##' @rdname "unchanged"
#register_preproc("random_vertical_flip", torchvision::transform_random_vertical_flip, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchRandomGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_grayscale", torchvision::transform_random_grayscale, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchRandomAffine
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_affine", torchvision::transform_random_affine, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchRandomPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_perspective", torchvision::transform_random_perspective, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchVflip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_vflip", torchvision::transform_vflip, packages = "torchvision", shapes_out = "unchanged") # nolint
#
##' @name PipeOpPreprocTorchRandomErasing
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_erasing", torchvision::transform_random_erasing, packages = "torchvision", shapes_out = "infer")
#
##' @name PipeOpPreprocTorchPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_perspective", torchvision::transform_perspective, packages = "torchvision", shapes_out = "infer") # nolint
#
##' @name PipeOpPreprocTorchRandomApply
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_apply", torchvision::transform_random_apply, packages = "torchvision", shapes_out = NULL) # nolint
#
##' @name PipeOpPreprocTorchRotate
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_rotate", torchvision::transform_rotate, packages = "torchvision", shapes_out = "infer") # nolint
#
##' @name PipeOpPreprocTorchCenterCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_center_crop", torchvision::transform_center_crop, packages = "torchvision", shapes_out = "infer") # nolint
#
##' @name PipeOpPreprocTorchRandomChoice
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_choice", torchvision::transform_random_choice, packages = "torchvision", shapes_out = NULL) # nolint
#
#
##' @name PipeOpPreprocTorchRandomCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("augment_random_crop", torchvision::transform_random_crop, packages = "torchvision", shapes_out = NULL) # nolint
#
