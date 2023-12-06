test_that("trafo_resize", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_resize",  size = c(3, 4)),
    shapes_in = list(c(16, 10, 10, 4), c(3, 4, 8))
  )
})

test_that("trafo_resize", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_resize",  size = c(3, 4)),
    shapes_in = list(c(16, 10, 10, 4), c(3, 4, 8))
  )
})

test_that("trafo_nop", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_nop"),
    shapes_in = list(c(1, 1))
  )
})

test_that("trafo_adjust_gamma", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_adjust_gamma", gamma = 0.2, gain = 2),
    shapes_in = list(c(4, 3, 10, 10))
  )
})

test_that("trafo_adjust_brightness", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_adjust_brightness", brightness_factor = 0.2),
    shapes_in = list(c(2, 3, 8, 8))
  )
})

test_that("trafo_adjust_hue", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_adjust_hue", hue_factor = 0.3),
    shapes_in = list(c(5, 3, 8, 8))
  )
})

test_that("trafo_adjust_saturation", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_adjust_saturation", saturation_factor = 2),
    shapes_in = list(c(2, 3, 8, 8))
  )
})

# not implemented
#test_that("trafo_grayscale", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("trafo_grayscale", num_output_channels = 3),
#    shapes_in = list(c(2, 3, 8, 8))
#  )
#})

test_that("trafo_rgb_to_grayscale", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_rgb_to_grayscale"),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("trafo_normalize", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_normalize", mean = -2, std = 3),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("trafo_pad", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_pad", padding = c(2, 3)),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

## Augmentation

test_that("augment_resized_crop", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_resized_crop", top = 1, left = 2, height = 5, width = 6, size = c(10, 11)),
    shapes_in = list(c(1, 3, 64, 64))
  )
})

test_that("augment_color_jitter", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_color_jitter"),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("augment_random_resized_crop", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_resized_crop", size = c(4, 5)),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("augment_random_order", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_order", transforms = list(
      function(x) torchvision::transform_resize(x, c(4, 5))
    )),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("augment_hflip", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_hflip"),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

#test_that("augment_random_rotation", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("augment_random_rotation", degrees = 20),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

test_that("augment_random_horizontal_flip", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_horizontal_flip"),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

#test_that("augment_linear_transformation", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("augment_linear_transformation",
#      transformation_matrix = torch_randn(rep(3 * 8 * 7, 2)), mean_vector = torch_randn(3 * 8 * 7)),
#    shapes_in = list(c(1, 3, 8, 7))
#  )
#})

test_that("augment_crop", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_crop", top = 2, left = 3, height = 10, width = 9),
    shapes_in = list(c(2, 3, 11, 9))
  )
})

test_that("augment_random_vertical_flip", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_vertical_flip"),
    shapes_in = list(c(3, 3, 8, 7))
  )
})

# not implemented for torch_tensor
#test_that("augment_random_grayscale", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("augment_random_vertical_flip"),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

#test_that("augment_random_perspective", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("augment_random_perspective"),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

test_that("augment_random_affine", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_affine", degrees = c(2, 70)),
    shapes_in = list(c(1, 3, 8, 8))
  )
})


test_that("augment_vflip", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_vflip"),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

#test_that("augment_random_erasing", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("augment_random_erasing"),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

# not implemented
#test_that("augment_perspective", {
#  autotest_pipeop_torch_preprocess(
#    obj = po("augment_perspective", startpoints = 3, endpoints = 4),
#    shapes_in = list(c(1, 3, 8, 8))
#  )
#})

test_that("augment_random_apply", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_apply", transforms = list(
      function(x) torchvision::transform_resize(x, c(4, 5))
    )),
    shapes_in = list(c(1, 3, 9, 18))
  )
})

test_that("augment_rotate", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_rotate", angle = 3),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("augment_center_crop", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_center_crop", size = c(6, 5)),
    shapes_in = list(c(6, 3, 10, 11))
  )
})

test_that("augment_random_choice", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_choice", transforms = list(
      identity, identity
    )),
    shapes_in = list(c(1, 3, 8, 8))
  )
})

test_that("augment_random_crop", {
  autotest_pipeop_torch_preprocess(
    obj = po("augment_random_crop", size = c(4, 4)),
    shapes_in = list(c(1, 3, 8, 8))
  )
})
