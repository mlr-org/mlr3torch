#' @title Lazy Preprocessing and Transformations
#' @description
#' Overview over all implemented preprocessing methods.
#' See [`PipeOpTaskPreprocTorch`] for more details on the class itself.
#' @section Overview:
#' * trafo_resize: Calls [`torchvision::transform_resize`]
#' @usage NULL
#' @format NULL
#' @name mlr_pipeops_preproc_torch_overview
#' @rawNamespace exportPattern("^PipeOpPreprocTorch")
#' @include PipeOpTaskPreprocTorch.R
NULL

table_preproc = function() {
  keys = names(mlr3torch_pipeops)
  keys = keys[grepl("^preproc_(?!torch).*", keys, perl = TRUE)]
  paste0("* ", keys)
}

#' @name PipeOpPreprocTorchTrafoResize
#' @rdname mlr_pipeops_preproc_torch_overview
register_preproc("trafo_resize", torchvision::transform_resize,
  packages = "torchvision",
  param_set = ps(
    size = p_uty(tags = c("train", "required")),
    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
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
  }
)



## Preprocessing:
#
##' @name PipeOpPreprocTorchAdjustGamma
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("adjust_gamma", torchvision::transform_adjust_gamma, packages = "torchvision"
#)
#
##' @name PipeOpPreprocTorchAdjustBrightness
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("adjust_brightness", torchvision::transform_adjust_brightness, packages = "torchvision")
#
##' @name PipeOpPreprocTorchAdjustHue
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("adjust_hue", torchvision::transform_adjust_hue, packages = "torchvision")
#
##' @name PipeOpPreprocTorchRandomCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_crop", torchvision::transform_random_crop, packages = "torchvision",
#  shapes_out = TRUE
#)
#
##' @name PipeOpPreprocTorchAdjustSaturation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("adjust_saturation", torchvision::transform_adjust_saturation, packages = "torchvision")
#
##' @name PipeOpPreprocTorchGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("grayscale", torchvision::transform_grayscale, packages = "torchvision")
##' @name PipeOpPreprocTorchRgdToGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("rgb_to_grayscale", torchvision::transform_rgb_to_grayscale, packages = "torchvision")
#
##' @name PipeOpPreprocTorchNormalize
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("normalize", torchvision::transform_normalize, packages = "torchvision")
#
##' @name PipeOpPreprocTorchPad
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("pad", torchvision::transform_pad, packages = "torchvision")
#
## Data Augmentation:
#
##' @name PipeOpPreprocTorchResizedCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("resized_crop", torchvision::transform_resized_crop, packages = "torchvision")
##' @name PipeOpPreprocTorchColorJitter
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("color_jitter", torchvision::transform_color_jitter, torchvision::transform_color_jitter, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomResizedCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_resized_crop", torchvision::transform_random_resized_crop, packages = "torchvision")
##' @name PipeOpPreprocTorchFiveCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("five_crop", torchvision::transform_five_crop, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomOrder
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_order", torchvision::transform_random_order, packages = "torchvision")
##' @name PipeOpPreprocTorchHflip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("hflip", torchvision::transform_hflip, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomRotation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_rotation", torchvision::transform_random_rotation, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomHorizontalFlip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_horizontal_flip", torchvision::transform_random_horizontal_flip, packages = "torchvision")
##' @name PipeOpPreprocTorchLinerTransformation
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("linear_transformation", torchvision::transform_linear_transformation, packages = "torchvision")
##' @name PipeOpPreprocTorchCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("crop", torchvision::transform_crop, packages = "torchvision")
##' @name PipeOpPreprocTorchTenCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("ten_crop", torchvision::transform_ten_crop, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomVerticalFlip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_vertical_flip", torchvision::transform_random_vertical_flip, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomGrayscale
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_grayscale", torchvision::transform_random_grayscale, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomAffine
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_affine", torchvision::transform_random_affine, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_perspective", torchvision::transform_random_perspective, packages = "torchvision")
##' @name PipeOpPreprocTorchVflip
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("vflip", torchvision::transform_vflip, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomErasing
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_erasing", torchvision::transform_random_erasing, packages = "torchvision")
##' @name PipeOpPreprocTorchPerspective
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("perspective", torchvision::transform_perspective, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomApply
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_apply", torchvision::transform_random_apply, packages = "torchvision")
##' @name PipeOpPreprocTorchRotate
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("rotate", torchvision::transform_rotate, packages = "torchvision")
##' @name PipeOpPreprocTorchCenterCrop
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("center_crop", torchvision::transform_center_crop, packages = "torchvision")
##' @name PipeOpPreprocTorchRandomChoice
##' @rdname mlr_pipeops_preproc_torch_overview
#register_preproc("random_choice", torchvision::transform_random_choice, packages = "torchvision")
