paramsets_image_trafo = Dictionary$new()


make_paramset_random_crop = function() {
  ps(
    size = p_uty(tags = c("train", "predict", "required")),
    padding = p_uty(default = NULL, tags = c("train", "predict")),
    pad_if_needed = p_lgl(default = FALSE, tags = c("train", "predict")),
    fill = p_uty(default = 0L, tags = c("train", "predict")),
    padding_mode = p_fct(default = "constant", tags = c("train", "predict"),
      levels = c("constant", "edge", "reflect", "symmetric")
    )
  )
}

make_paramset_center_crop = function() {
  ps(
    size = p_uty(tags = c("train", "predict", "required"))
  )

}

make_paramset_hflip = function() {
  ps()
}

make_paramset_adjust_gamma = function() {
  ps(
    gamma = p_dbl(tags = c("train", "predict", "required")),
    gain = p_dbl(default = 1, tags = c("train", "predict"))
  )

}

make_paramset_random_order = function() {
  ps(
    transforms = p_uty(tags = c("train", "predict", "required"))
  )
}

make_paramset_adjust_brightness = function() {
  ps(
    brightness_factor = p_dbl(lower = 0, tags = c("train", "predict", "required"))
  )
}

make_paramset_pad = function() {
  ps(
    padding = p_uty(tags = c("train", "predict", "required")),
    fill = p_uty(default = 0L, tags = c("train", "predict")),
    padding_mode = p_fct(default = "constant", tags = c("train", "predict"),
      levels = c("constant", "edge", "reflect", "symmetric")
    )
  )

}

make_paramset_random_affine = function() {
  ps(
    degrees = p_uty(tags = c("train", "predict", "required")),
    translate = p_uty(default = NULL, tags = c("train", "predict")),
    scale = p_uty(default = NULL, tags = c("train", "predict")),
    shear = p_uty(default = NULL, tags = c("train", "predict")),
    resample = p_int(default = 0L, tags = c("train", "predict")),
    fillcolor = p_uty(default = 0L, tags = c("train", "predict"))
  )
}

make_paramset_affine = function() {
  ps(
    angle = p_uty(tags = c("train", "predict", "required")),
    translate = p_uty(tags = c("train", "predict")),
    scale = p_uty(tags = c("train", "predict")),
    shear = p_uty(tags = c("train", "predict")),
    resample = p_int(default = 0L, tags = c("train", "predict")),
    fillcolor = p_uty(default = NULL, tags = c("train", "predict"))
  )

}

make_paramset_random_rotation = function() {
  ps(
    degrees = p_uty(tags = c("train", "predict", "required")),
    resample = p_int(default = FALSE, tags = c("train", "predict"), special_vals = list(FALSE)),
    expand = p_lgl(default = FALSE, tags = c("train", "predict")),
    center = p_uty(default = NULL, tags = c("train", "predict")),
    fill = p_uty(default = NULL, tags = c("train", "predict"))
  )
}

make_paramset_vflip = function() {
  ps()
}

make_paramset_random_resized_crop = function() {
  ps(
    size = p_uty(tags = c("train", "predict", "required")),
    scale = p_uty(default = c(0.08, 1), tags = c("train", "predict")),
    ratio = p_uty(default = c(3/4, 4/3), tags = c("train", "predict")),
    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
      tags = c("train", "predict")
    )
  )
}

make_paramset_crop = function() {
  ps(
    top = p_int(tags = c("train", "predict", "required")),
    left = p_int(tags = c("train", "predict", "required")),
    height = p_int(tags = c("train", "predict", "required")),
    width = p_int(tags = c("train", "predict", "required"))
  )
}

make_paramset_resized_crop = function() {
  ps(
    top = p_int(tags = c("train", "predict", "required")),
    left = p_int(tags = c("train", "predict", "required")),
    height = p_int(tags = c("train", "predict", "required")),
    width = p_int(tags = c("train", "predict", "required")),
    size = p_uty(tags = c("train", "predict", "required")),
    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
      tags = c("train", "predict")
    )
  )
}

make_paramset_random_choice = function() {
  ps(
    transforms = p_uty(tags = c("train", "predict", "required"))
  )

}

make_paramset_resize = function() {
  ps(
    size = p_uty(tags = c("train", "predict", "required")),
    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
      tags = c("train", "predict")
    )
  )
}

make_paramset_rgb_to_grayscale = function() {
  ps()
}


make_paramset_adjust_saturation = function() {
  ps(
    saturation_factor = p_dbl(tags = c("train", "predict", "required"))
  )
}

make_paramset_linear_transformation = function() {
  ps(
    transformation_matrix = p_uty(tags = c("train", "predict", "required")),
    mean_vector = p_uty(tags = c("train", "predict", "required"))
  )
}

make_paramset_random_vertical_flip = function() {
  ps(
    p = p_dbl(lower = 0, upper = 1, default = 1, tags = c("train", "predict"))
  )
}

make_paramset_random_horizontal_flip = function() {
  ps(
    p = p_dbl(lower = 0, upper = 1, default = 1, tags = c("train", "predict"))

  )
}

make_paramset_color_jitter = function() {
  ps(
    brightness = p_uty(default = 0, tags = c("train", "predict")),
    contrast = p_uty(default = 0, tags = c("train", "predict")),
    saturation = p_uty(default = 0, tags = c("train", "predict")),
    hue = p_uty(default = 0, tags = c("train", "predict"))
  )
}

make_paramset_adjust_contrast = function() {
  ps(
    contrast_factor = p_dbl(lower = 0, tags = c("train", "predict", "required"))
  )
}

make_paramset_rotate = function() {
  ps(
    angle = p_uty(tags = c("train", "predict", "required")),
    resample = p_int(default = 0L, tags = c("train", "predict")),
    expand = p_lgl(default = FALSE, tags = c("train", "predict")),
    center = p_uty(default = NULL, tags = c("train", "predict")),
    fill = p_uty(default = NULL, tags = c("train", "predict"))
  )
}

make_paramset_adjust_hue = function() {
  ps(
    hue_factor = p_dbl(lower = 0.5, upper = 0.5, tags = c("train", "predict", "required"))
  )
}

make_paramset_normalize = function() {
  ps(
    mean = p_uty(tags = c("train", "predict", "required")),
    std = p_uty(tags = c("train", "predict", "require")),
    inplace = p_lgl(default = FALSE, tags = c("train", "predict"))
  )
}

make_paramset_random_apply = function() {
  ps(
    transforms = p_uty(tags = c("train", "predict", "required")),
    p = p_dbl(lower = 0, upper = 1, default = 0.5, tags = c("train", "predict"))
  )
}

make_paramset_to_tensor = function() {
  ps()
}

paramsets_image_trafo$add("random_crop", make_paramset_random_crop)
paramsets_image_trafo$add("center_crop", make_paramset_center_crop)
paramsets_image_trafo$add("hflip", make_paramset_hflip)
paramsets_image_trafo$add("adjust_gamma", make_paramset_adjust_gamma)
paramsets_image_trafo$add("random_order", make_paramset_random_order)
paramsets_image_trafo$add("adjust_brightness", make_paramset_adjust_brightness)
paramsets_image_trafo$add("pad", make_paramset_pad)
paramsets_image_trafo$add("random_affine", make_paramset_random_affine)
paramsets_image_trafo$add("affine", make_paramset_affine)
paramsets_image_trafo$add("random_rotation", make_paramset_random_rotation)
paramsets_image_trafo$add("vflip", make_paramset_vflip)
paramsets_image_trafo$add("random_resized_crop", make_paramset_random_resized_crop)
paramsets_image_trafo$add("crop", make_paramset_crop)
paramsets_image_trafo$add("resized_crop", make_paramset_resized_crop)
paramsets_image_trafo$add("random_choice", make_paramset_random_choice)
paramsets_image_trafo$add("resize", make_paramset_resize)
paramsets_image_trafo$add("rgb_to_grayscale", make_paramset_rgb_to_grayscale)
paramsets_image_trafo$add("adjust_saturation", make_paramset_adjust_saturation)
paramsets_image_trafo$add("linear_transformation", make_paramset_linear_transformation)
paramsets_image_trafo$add("random_vertical_flip", make_paramset_random_vertical_flip)
paramsets_image_trafo$add("random_horizontal_flip", make_paramset_random_horizontal_flip)
paramsets_image_trafo$add("color_jitter", make_paramset_color_jitter)
paramsets_image_trafo$add("adjust_contrast", make_paramset_adjust_contrast)
paramsets_image_trafo$add("rotate", make_paramset_rotate)
paramsets_image_trafo$add("adjust_hue", make_paramset_adjust_hue)
paramsets_image_trafo$add("normalize", make_paramset_normalize)
paramsets_image_trafo$add("random_apply", make_paramset_random_apply)
paramsets_image_trafo$add("to_tensor", make_paramset_to_tensor)
