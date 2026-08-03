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
    obj = po("augment_crop", top = 2, left = 3, height = 10, width = 7),
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

test_that("shape inference matches the operator", {
  expect_shapes_out_preproc("trafo_resize", list(size = c(8, 12)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_resize", list(size = 8), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_resize", list(size = 8), c(2, 3, 21, 16))
  expect_shapes_out_preproc("trafo_pad", list(padding = 2), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_pad", list(padding = c(1, 2)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_pad", list(padding = c(1, 2, 3, 4)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("augment_crop", list(top = 3, left = 4, height = 10, width = 12), c(2, 3, 16, 20))
  expect_shapes_out_preproc("augment_crop", list(top = 10, left = 10, height = 7, width = 11), c(2, 3, 16, 20))
  expect_shapes_out_preproc("augment_center_crop", list(size = 8), c(2, 3, 16, 20))
  expect_equal(po("augment_center_crop", size = 32)$shapes_out(list(c(2L, 3L, 16L, 20L)),
    stage = "train")[[1L]], c(2L, 3L, NA, NA))
  expect_shapes_out_preproc("augment_resized_crop", list(top = 1, left = 1, height = 8, width = 8, size = c(4, 4)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_grayscale", list(num_output_channels = 1), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_grayscale", list(num_output_channels = 3), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_rgb_to_grayscale", list(), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_adjust_gamma", list(gamma = 0.5), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_adjust_brightness", list(brightness_factor = 0.5), c(2, 3, 16, 20))
  expect_shapes_out_preproc("augment_color_jitter", list(), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_reshape", list(shape = c(-1, 3, 320)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_nop", list(), c(2, 3, 16, 20))
})

test_that("shape inference follows torchvision, including where it misbehaves", {
  shapes_out = function(id, pv, shape) {
    obj = po(id)
    if (length(pv)) obj$param_set$set_values(.values = pv)
    obj$shapes_out(list(as.integer(shape)), stage = "train")[[1L]]
  }
  # a crop that leaves the image is rejected, because `transform_crop()` clamps instead of
  # padding, see the FIXME in R/preprocess.R
  expect_error(shapes_out("augment_crop", list(top = 10, left = 10, height = 20, width = 20),
    c(2, 3, 16, 20)), "clamps instead of padding", fixed = TRUE)
  expect_error(shapes_out("augment_crop", list(top = 20, left = 4, height = 5, width = 5),
    c(2, 3, 16, 20)), "clamps instead of padding", fixed = TRUE)
  # a crop that fits is unaffected
  expect_equal(shapes_out("augment_crop", list(top = 10, left = 10, height = 7, width = 11),
    c(2, 3, 16, 20)), c(2L, 3L, 7L, 11L))
  # `transform_center_crop()` is rejected when it would pad a non-square size
  expect_error(shapes_out("augment_center_crop", list(size = c(24, 30)), c(2, 3, 16, 20)),
    "pads a non-square size incorrectly", fixed = TRUE)
  # `transform_resized_crop()` crops and then resizes, and a scalar `size` preserves the aspect
  # ratio of the cropped image
  expect_equal(shapes_out("augment_resized_crop",
    list(top = 1, left = 1, height = 8, width = 16, size = 4), c(2, 3, 16, 20)), c(2L, 3L, 4L, 8L))
  # the flips read no dimension, so an unknown or unusual channel count is fine
  expect_equal(shapes_out("augment_hflip", list(), c(2, 4, 16, 20)), c(2L, 4L, 16L, 20L))
  expect_equal(shapes_out("augment_random_horizontal_flip", list(), c(2, 1, 16, 20)), c(2L, 1L, 16L, 20L))
  expect_equal(shapes_out("augment_vflip", list(), c(2, NA, 16, 20)), c(2L, NA, 16L, 20L))
  # `transform_color_jitter()` truncates to three channels once `hue` is active
  expect_equal(shapes_out("augment_color_jitter", list(hue = 0.2), c(2, 4, 8, 10)), c(2L, 3L, 8L, 10L))
  expect_equal(shapes_out("augment_color_jitter", list(), c(2, 4, 8, 10)), c(2L, 4L, 8L, 10L))
})

test_that("resize_extent requires 'size' to have 1 or 2 values", {
  # the operators check this in their parameter set as well, so the helper is called directly
  expect_error(resize_extent(c(16L, 20L), c(8L, 12L, 99L), "po"), "1 or 2 values, but it has 3")
  expect_error(resize_extent(c(16L, 20L), integer(0), "po"), "1 or 2 values, but it has 0")
  expect_equal(resize_extent(c(16L, 20L), c(8L, 12L), "po"), c(8L, 12L))
  expect_equal(resize_extent(c(16L, 20L), 8L, "po"), c(8L, 10L))
})
