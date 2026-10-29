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

test_that("operators applied to the whole batch give the same result as applying them to every image", {
  per_image = function(name, x, args) {
    fn = getFromNamespace(name, "torchvision")
    torch_stack(lapply(seq_len(x$shape[1L]), function(i) do.call(fn, c(list(x[i, ..]), args))), dim = 1L)
  }
  via_pipeop = function(id, x, args) {
    task = as_task_regr(data.table(x = as_lazy_tensor(x), y = as.numeric(seq_len(x$shape[1L]))), target = "y")
    obj = do.call(po, c(list(id), args))
    materialize(obj$train(list(task))[[1L]]$data(cols = "x")$x, rbind = TRUE)
  }
  inputs = list(
    image = torch_rand(5, 3, 8, 10),
    channelless = torch_rand(5, 8, 10),
    five_dim = torch_rand(5, 2, 3, 8, 10)
  )
  cases = list(
    list("trafo_adjust_gamma", "transform_adjust_gamma", list(gamma = 0.5), c("image", "channelless", "five_dim")),
    list("trafo_adjust_gamma", "transform_adjust_gamma", list(gamma = 2, gain = 0.8), "image"),
    list("trafo_grayscale", "transform_grayscale", list(num_output_channels = 1L), "image"),
    list("trafo_grayscale", "transform_grayscale", list(num_output_channels = 3L), "image"),
    list("trafo_rgb_to_grayscale", "transform_rgb_to_grayscale", list(), c("image", "five_dim")),
    list("trafo_pad", "transform_pad", list(padding = 2L), c("image", "channelless", "five_dim")),
    list("trafo_pad", "transform_pad", list(padding = c(1L, 2L, 3L, 4L), fill = 0.5), c("image", "channelless")),
    list("trafo_pad", "transform_pad", list(padding = 2L, padding_mode = "edge"), c("image", "channelless", "five_dim")),
    list("trafo_pad", "transform_pad", list(padding = 2L, padding_mode = "reflect"), c("image", "channelless", "five_dim")),
    list("augment_hflip", "transform_hflip", list(), c("image", "channelless", "five_dim")),
    list("augment_vflip", "transform_vflip", list(), c("image", "channelless", "five_dim")),
    list("augment_crop", "transform_crop", list(top = 2L, left = 3L, height = 4L, width = 5L),
      c("image", "channelless", "five_dim")),
    list("augment_center_crop", "transform_center_crop", list(size = 4L), c("image", "channelless", "five_dim")),
    # pads, as the images are smaller than the size
    list("augment_center_crop", "transform_center_crop", list(size = 12L), c("image", "channelless")),
    list("augment_resized_crop", "transform_resized_crop",
      list(top = 1L, left = 2L, height = 6L, width = 6L, size = 4L), c("image", "five_dim")),
    list("augment_resized_crop", "transform_resized_crop",
      list(top = 1L, left = 1L, height = 7L, width = 5L, size = c(6L, 4L), interpolation = 0L), "image"),
    list("augment_resized_crop", "transform_resized_crop",
      list(top = 1L, left = 1L, height = 7L, width = 5L, size = c(6L, 4L), interpolation = 3L), "image")
  )
  for (case in cases) {
    for (input in case[[4L]]) {
      x = inputs[[input]]
      expected = per_image(case[[2L]], x, case[[3L]])
      out = via_pipeop(case[[1L]], x, case[[3L]])
      label = sprintf("%s (%s) on input '%s'", case[[1L]], paste(names(case[[3L]]), collapse = ", "), input)
      expect_equal(out$shape, expected$shape, label = label)
      expect_true(torch_allclose(out, expected, atol = 1e-6), label = label)
    }
  }
})

test_that("trafo_normalize normalizes a batch like torchvision::transform_normalize() does every image", {
  x = torch_randn(4, 3, 5, 6)
  mean = c(0.1, -0.2, 0.3)
  std = c(1, 2, 0.5)
  expected = torch_stack(lapply(1:4, function(i) torchvision::transform_normalize(x[i, ..], mean, std)), dim = 1L)
  expect_true(torch_allclose(normalize_batch(x, mean, std), expected))

  task = as_task_regr(data.table(x = as_lazy_tensor(x), y = rnorm(4)), target = "y")
  task_out = po("trafo_normalize", mean = mean, std = std)$train(list(task))[[1L]]
  expect_true(torch_allclose(materialize(task_out$data(cols = "x")$x, rbind = TRUE), expected))

  # without spatial dimensions, every column is a channel
  z = torch_randn(5, 3)
  expect_true(torch_allclose(normalize_batch(z, mean, std), (z - torch_tensor(mean)) / torch_tensor(std)))
  # a single value for all channels
  expect_true(torch_allclose(normalize_batch(x, 1, 2), (x - 1) / 2))

  # videos (N, T, C, H, W): the channels are the third-to-last dimension, like for torchvision per element
  v = torch_randn(2, 4, 3, 5, 6)
  expected_v = torch_stack(lapply(1:2, function(i) torchvision::transform_normalize(v[i, ..], mean, std)), dim = 1L)
  expect_true(torch_allclose(normalize_batch(v, mean, std), expected_v))
  # sequences (N, C, L): the channels are the second dimension
  s3 = torch_randn(2, 3, 7)
  expect_true(torch_allclose(normalize_batch(s3, mean, std),
    (s3 - torch_tensor(mean)$view(c(1, 3, 1))) / torch_tensor(std)$view(c(1, 3, 1))))
  expect_error(normalize_batch(z, 0, c(1, 0, 1)), "must not contain zeros")
})

test_that("trafo_normalize does not truncate mean and std for integer inputs", {
  mean = c(127.5, 100.2, 30.7)
  std = c(60.3, 2.5, 1.5)
  x = torch_randint(0, 256, c(4, 3, 5, 6))
  expected = (x$to(dtype = torch_float()) - torch_tensor(mean)$view(c(1, 3, 1, 1))) /
    torch_tensor(std)$view(c(1, 3, 1, 1))
  for (dtype in list(torch_long(), torch_int(), torch_uint8())) {
    out = normalize_batch(x$to(dtype = dtype), mean, std)
    expect_true(out$dtype == torch_float())
    expect_true(torch_allclose(out, expected))
  }

  task = as_task_regr(data.table(x = as_lazy_tensor(x$to(dtype = torch_long())), y = rnorm(4)), target = "y")
  task_out = po("trafo_normalize", mean = mean, std = std)$train(list(task))[[1L]]
  expect_true(torch_allclose(materialize(task_out$data(cols = "x")$x, rbind = TRUE), expected))
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
  expect_shape_inference("trafo_resize", list(size = c(8, 12)), c(2, 3, 16, 20))
  expect_shape_inference("trafo_resize", list(size = 8), c(2, 3, 16, 20))
  expect_shape_inference("trafo_resize", list(size = 8), c(2, 3, 21, 16))
  expect_shape_inference("trafo_pad", list(padding = 2), c(2, 3, 16, 20))
  expect_shape_inference("trafo_pad", list(padding = c(1, 2)), c(2, 3, 16, 20))
  expect_shape_inference("trafo_pad", list(padding = c(1, 2, 3, 4)), c(2, 3, 16, 20))
  expect_shape_inference("augment_crop", list(top = 3, left = 4, height = 10, width = 12), c(2, 3, 16, 20))
  expect_shape_inference("augment_crop", list(top = 10, left = 10, height = 7, width = 11), c(2, 3, 16, 20))
  expect_shape_inference("augment_center_crop", list(size = 8), c(2, 3, 16, 20))
  expect_equal(po("augment_center_crop", size = 32)$shapes_out(list(c(2L, 3L, 16L, 20L)),
    stage = "train")[[1L]], c(2L, 3L, NA, NA))
  expect_shape_inference("augment_resized_crop", list(top = 1, left = 1, height = 8, width = 8, size = c(4, 4)), c(2, 3, 16, 20))
  expect_shape_inference("trafo_grayscale", list(num_output_channels = 1), c(2, 3, 16, 20))
  expect_shape_inference("trafo_grayscale", list(num_output_channels = 3), c(2, 3, 16, 20))
  expect_shape_inference("trafo_rgb_to_grayscale", list(), c(2, 3, 16, 20))
  expect_shape_inference("trafo_adjust_gamma", list(gamma = 0.5), c(2, 3, 16, 20))
  expect_shape_inference("trafo_adjust_brightness", list(brightness_factor = 0.5), c(2, 3, 16, 20))
  expect_shape_inference("augment_color_jitter", list(), c(2, 3, 16, 20))
  expect_shape_inference("trafo_adjust_saturation", list(saturation_factor = 0.5), c(2, 3, 16, 20))
  expect_shape_inference("trafo_adjust_hue", list(hue_factor = 0.2), c(2, 3, 16, 20))
  expect_shape_inference("trafo_normalize", list(mean = 0, std = 1), c(2, 3, 16, 20))
  expect_shape_inference("augment_hflip", list(), c(2, 3, 16, 20))
  expect_shape_inference("augment_vflip", list(), c(2, 3, 16, 20))
  expect_shape_inference("augment_random_horizontal_flip", list(p = 1), c(2, 3, 16, 20))
  expect_shape_inference("augment_random_vertical_flip", list(p = 1), c(2, 3, 16, 20))
  expect_shape_inference("trafo_reshape", list(shape = c(-1, 3, 320)), c(2, 3, 16, 20))
  expect_shape_inference("trafo_nop", list(), c(2, 3, 16, 20))
})

# the preprocessing operators all work on (batch, channels, height, width) images
gen_image_shape = function() {
  function() c(sample(1:3, 1L), 3L, size(2L, 6L, 16L))
}

# The parameter draws of the preprocessing operators, one entry per operator. The test below
# enrols all of them, so an operator without an entry here is checked only for declaring an
# unknown output shape.
preproc_inference_specs = function() {
  list(
    trafo_resize = function() list(size = if (sample(c(TRUE, FALSE), 1L)) size(1L) else size(2L)),
    trafo_pad = function() list(padding = size(sample(c(1L, 2L, 4L), 1L), 0L, 3L)),
    trafo_grayscale = function() list(num_output_channels = sample(1:3, 1L)),
    trafo_rgb_to_grayscale = function() list(),
    trafo_adjust_gamma = function() list(gamma = 0.5),
    trafo_adjust_brightness = function() list(brightness_factor = 0.5),
    trafo_adjust_saturation = function() list(saturation_factor = 0.5),
    trafo_adjust_hue = function() list(hue_factor = 0.2),
    trafo_normalize = function() list(mean = 0, std = 1),
    trafo_nop = function() list(),
    augment_color_jitter = function() list(hue = sample(c(0, 0.2), 1L)),
    augment_hflip = function() list(),
    augment_vflip = function() list(),
    augment_random_horizontal_flip = function() list(p = 1),
    augment_random_vertical_flip = function() list(p = 1),
    augment_crop = function() list(top = sample(1:4, 1L), left = sample(1:4, 1L),
      height = size(1L, 1L, 3L), width = size(1L, 1L, 3L)),
    augment_center_crop = function() list(size = size(1L, 2L, 6L)),
    augment_resized_crop = function() list(top = 1L, left = 1L, height = size(1L, 2L, 8L),
      width = size(1L, 2L, 8L), size = if (sample(c(TRUE, FALSE), 1L)) size(1L, 2L, 6L) else size(2L, 2L, 6L))
  )
}

test_that("every preprocessing PipeOp agrees with its function for random shapes and parameters", {
  specs = preproc_inference_specs()
  for (id in names(specs)) {
    expect_shape_inference(id, params = specs[[id]], generators = gen_image_shape())
  }
})

test_that("the remaining preprocessing PipeOps declare an unknown output shape", {
  # declaring `NULL`, i.e. "unknown", is always safe: it never claims a size the tensor does not
  # have. This checks that the operators without a sampler in `preproc_inference_specs()` really do
  # abstain rather than being left unverified by an oversight.
  specs = preproc_inference_specs()
  unknown = Filter(function(key) {
    obj = suppressWarnings(try(po(key), silent = TRUE))
    !inherits(obj, "try-error") && inherits(obj, "PipeOpTaskPreprocTorch") && key %nin% names(specs)
  }, mlr_pipeops$keys())
  # if this filter ever stops matching, the loop below would silently check nothing
  expect_true(length(unknown) >= 5L)

  for (id in unknown) {
    obj = po(id)
    # operators with unset required parameters cannot be asked for their shapes
    shape = tryCatch(obj$shapes_out(list(c(NA, 3L, 8L, 8L)), stage = "train")[[1L]],
      error = function(e) "skip")
    if (identical(shape, "skip")) next
    expect_true(is.null(shape), label = sprintf("%s declares an unknown output shape", id))
  }
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
