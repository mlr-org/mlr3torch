#' @title Reflections mechanism for torch
#'
#' @details
#' Used to store / extend available hyperparameter levels for options used throughout torch,
#' e.g. the available 'loss' for a given Learner.
#'
#' @format [environment].
#' @export
torch_reflections = new.env(parent = emptyenv())

local({

  torch_reflections$activation = c(
    "elu", "hardshrink", "hardsigmoid", "hardtanh", "hardswish", "leaky_relu", "log_sigmoid",
    "prelu", "relu", "relu6", "rrelu", "selu", "sigmoid",
    "softplus", "softshrink", "softsign", "tanh", "tanhshrink", "threshold", "glu"
  )

  torch_reflections$normalization = c("batch_norm")

  torch_reflections$image_trafos = c(
    "random_crop",
    "center_crop",
    "hflip",
    "adjust_gamma",
    "random_order",
    "adjust_brightness",
    "pad",
    "random_affine",
    "affine",
    "random_rotation",
    "vflip",
    "random_resized_crop",
    "crop",
    "resized_crop",
    "random_choice",
    "resize",
    "rgb_to_grayscale",
    "adjust_saturation",
    "linear_transformation",
    "random_vertical_flip",
    "random_horizontal_flip",
    "color_jitter",
    "adjust_contrast",
    "rotate",
    "adjust_hue",
    "normalize",
    "random_apply",
    "to_tensor"
  )
  torch_reflections$callback_steps = callback_steps = c(
    "start",
    "before_train_epoch",
    "before_train_batch",
    "after_train_batch",
    "before_valid_epoch",
    "before_valid_batch",
    "after_valid_batch",
    "after_valid_epoch",
    "end"
  )
})
