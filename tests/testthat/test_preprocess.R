test_that("trafo_resize", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_resize",  size = c(3, 4)),
    shapes_in = list(c(16, 10, 10, 4), c(3, 4, 8)),
    deterministic = TRUE
  )
})

test_that("trafo_resize", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_resize",  size = c(3, 4)),
    shapes_in = list(c(16, 10, 10, 4), c(3, 4, 8)),
    deterministic = TRUE
  )
})

test_that("trafo_nop", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_nop"),
    shapes_in = list(c(5, 1)),
    deterministic = TRUE
  )
})

test_that("trafo_adjust_gamma", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_adjust_gamma", gamma = 0.2, gain = 2),
    shapes_in = list(c(4, 3, 10, 10)),
    deterministic = TRUE
  )
})

test_that("trafo_adjust_brightness", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_adjust_brightness", brightness_factor = 0.2),
    shapes_in = list(c(3, 3, 8, 8)),
    deterministic = TRUE
  )
})

test_that("trafo_adjust_hue", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_adjust_hue", hue_factor = 0.3),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

test_that("trafo_adjust_saturation", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_adjust_saturation", saturation_factor = 2),
    shapes_in = list(c(2, 3, 8, 8)),
    deterministic = TRUE
  )
})

# not implemented
#test_that("trafo_grayscale", {
#  expect_pipeop_torch_preprocess(
#    obj = po("trafo_grayscale", num_output_channels = 3),
#    shapes_in = list(c(2, 3, 8, 8))
#  )
#})

test_that("trafo_rgb_to_grayscale", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_rgb_to_grayscale"),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

test_that("trafo_normalize", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_normalize", mean = -2, std = 3),
    shapes_in = list(c(4, 3, 8, 8)),
    deterministic = TRUE
  )
})

test_that("trafo_pad", {
  expect_pipeop_torch_preprocess(
    obj = po("trafo_pad", padding = c(2, 3)),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

## Augmentation

test_that("augment_resized_crop", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_resized_crop", top = 1, left = 2, height = 5, width = 6, size = c(10, 11)),
    shapes_in = list(c(5, 3, 64, 64)),
    deterministic = TRUE
  )
})

test_that("augment_color_jitter", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_color_jitter"),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

test_that("augment_random_resized_crop", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_resized_crop", size = c(4, 5)),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
})

test_that("augment_random_order", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_order", transforms = list(
      function(x) torchvision::transform_resize(x, c(4, 5)),
      function(x) torchvision::transform_resize(x, c(4, 5)) + 1
    )),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_order", transforms = list(
      function(x) x
    )),
    shapes_in = list(c(5, 2)),
    deterministic = FALSE
  )
})

test_that("augment_hflip", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_hflip"),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

#test_that("augment_random_rotation", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_random_rotation", degrees = 20),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

test_that("augment_random_horizontal_flip", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_horizontal_flip"),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
})

#test_that("augment_linear_transformation", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_linear_transformation",
#      transformation_matrix = torch_randn(rep(3 * 8 * 7, 2)), mean_vector = torch_randn(3 * 8 * 7)),
#    shapes_in = list(c(1, 3, 8, 7))
#  )
#})

test_that("augment_crop", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_crop", top = 2, left = 3, height = 10, width = 9),
    shapes_in = list(c(5, 3, 11, 9)),
    deterministic = TRUE
  )
})

test_that("augment_random_vertical_flip", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_vertical_flip"),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
})

# not implemented for torch_tensor
#test_that("augment_random_grayscale", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_random_vertical_flip"),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

#test_that("augment_random_perspective", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_random_perspective"),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

test_that("augment_random_affine", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_affine", degrees = c(2, 70)),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
})


test_that("augment_vflip", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_vflip"),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

#test_that("augment_random_erasing", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_random_erasing"),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

# not implemented
#test_that("augment_perspective", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_perspective", startpoints = 3, endpoints = 4),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

# FIXME: here the rowwise parameter is problematic
#test_that("augment_random_apply", {
#  expect_pipeop_torch_preprocess(
#    obj = po("augment_random_apply", transforms = list(
#      function(x) torchvision::transform_resize(x, c(4, 5)),
#      function(x) torchvision::transform_resize(x, c(4, 5)) + 1
#    )),
#    shapes_in = list(c(5, 3, 9, 18)),
#    deterministic = FALSE
#  )
#})

test_that("augment_rotate", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_rotate", angle = 3),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = TRUE
  )
})

test_that("augment_center_crop", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_center_crop", size = c(6, 5)),
    shapes_in = list(c(6, 3, 10, 11)),
    deterministic = FALSE
  )
})

test_that("augment_random_choice", {
  # needs dev version of torchvision
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_choice", transforms = list(
      identity, function(x) x + 1
    )),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
})

test_that("augment_random_crop", {
  expect_pipeop_torch_preprocess(
    obj = po("augment_random_crop", size = c(4, 4)),
    shapes_in = list(c(5, 3, 8, 8)),
    deterministic = FALSE
  )
})
