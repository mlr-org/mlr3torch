test_that("pooling shapes match the operator, including dilation and ceil_mode", {
  # `dilation` enters the output extent, and with `ceil_mode` torch drops a pooling window that
  # would start inside the right-hand padding
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 2, stride = 1, dilation = 2), c(2, 2, 8, 8))
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 3, stride = 2, dilation = 3, padding = 1), c(2, 2, 16, 20))
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 2, stride = 3, ceil_mode = TRUE), c(2, 2, 6, 6))
  expect_shapes_out_torch("nn_max_pool1d", list(kernel_size = 2, stride = 2, padding = 1, ceil_mode = TRUE), c(2, 2, 5))
  expect_shapes_out_torch("nn_avg_pool2d", list(kernel_size = 2, stride = 2, padding = 1, ceil_mode = TRUE), c(2, 2, 5, 5))
  expect_shapes_out_torch("nn_avg_pool2d", list(kernel_size = 3, stride = 2, padding = 1), c(2, 3, 16, 20))
  expect_shapes_out_torch("nn_avg_pool1d", list(kernel_size = 2), c(2, 3, 17))
  expect_shapes_out_torch("nn_adaptive_avg_pool2d", list(output_size = c(2, 3)), c(2, 3, 16, 20))
})

test_that("shape inference of the neural network operators matches the operator", {
  expect_shapes_out_torch("nn_conv2d", list(out_channels = 5, kernel_size = 3, stride = 2, padding = 1), c(2, 3, 17, 19))
  expect_shapes_out_torch("nn_conv2d", list(out_channels = 4, kernel_size = 3, dilation = 2), c(2, 3, 16, 16))
  expect_shapes_out_torch("nn_conv1d", list(out_channels = 5, kernel_size = 3), c(2, 3, 17))
  expect_shapes_out_torch("nn_conv_transpose2d", list(out_channels = 5, kernel_size = 3, stride = 2), c(2, 3, 8, 9))
  expect_shapes_out_torch("nn_linear", list(out_features = 3), c(2, 7, 16))
  expect_shapes_out_torch("nn_layer_norm", list(dims = 2), c(2, 4, 7, 16))
  expect_shapes_out_torch("nn_batch_norm2d", list(), c(2, 3, 8, 8))
  expect_shapes_out_torch("nn_flatten", list(start_dim = 2, end_dim = 3), c(2, 4, 6, 8))
  expect_shapes_out_torch("nn_reshape", list(shape = c(-1, 24)), c(2, 4, 6))
  expect_shapes_out_torch("nn_unsqueeze", list(dim = 2), c(2, 4, 6))
  expect_shapes_out_torch("nn_unsqueeze", list(dim = -1), c(2, 4, 6))
  expect_shapes_out_torch("nn_squeeze", list(dim = 3), c(2, 4, 1, 6))
  expect_shapes_out_torch("nn_squeeze", list(), c(2, 4, 1, 6))
  expect_shapes_out_torch("nn_glu", list(dim = 2), c(2, 4, 6))
  expect_shapes_out_torch("nn_geglu", list(), c(2, 4, 6))
  expect_shapes_out_torch("nn_dropout", list(p = 0.5), c(2, 4, 6))
  expect_shapes_out_torch("nn_softmax", list(dim = 2), c(2, 4, 6))
  expect_shapes_out_torch("nn_ft_cls", list(initialization = "uniform"), c(2, 7, 16))
  expect_shapes_out_torch("nn_merge_sum", list(), c(2, 4, 6), n_in = 2L)
  expect_shapes_out_torch("nn_merge_prod", list(), c(2, 4, 6), n_in = 2L)
  expect_shapes_out_torch("nn_merge_cat", list(dim = 2), c(2, 4, 6), n_in = 2L)
  expect_shapes_out_torch("nn_head", list(), c(2, 16), task = tsk("iris"))
})

test_that("nn_squeeze accepts several dimensions", {
  # the module loops over the dimensions, so the shape inference has to allow a vector too
  expect_shapes_out_torch("nn_squeeze", list(dim = c(2L, 3L)), c(4, 1, 1, 8))
})

test_that("nn_ft_transformer_block returns one token per queried index", {
  # `query_idx` may select several tokens, and each of them appears in the output
  pv = list(attention_n_heads = 2, attention_dropout = 0, ffn_d_hidden_multiplier = 2,
    ffn_dropout = 0, residual_dropout = 0, attention_initialization = "kaiming",
    ffn_activation = nn_reglu, attention_normalization = nn_layer_norm,
    ffn_normalization = nn_layer_norm, attention_bias = TRUE, ffn_bias_first = TRUE,
    ffn_bias_second = TRUE, prenormalization = TRUE, is_first_layer = TRUE)
  expect_shapes_out_torch("nn_ft_transformer_block", c(pv, list(query_idx = 1L)), c(2, 5, 8))
  expect_shapes_out_torch("nn_ft_transformer_block", c(pv, list(query_idx = c(1L, 2L))), c(2, 5, 8))
  expect_shapes_out_torch("nn_ft_transformer_block", c(pv, list(query_idx = NULL)), c(2, 5, 8))
})

test_that("shape inference of the preprocessing operators matches the operator", {
  # These wrap torchvision, whose behaviour is not always what one would expect -- see the
  # comments in R/preprocess.R. `trafo_resize` with a single `size` matches the shorter side and
  # preserves the aspect ratio, so the output is not square.
  expect_shapes_out_preproc("trafo_resize", list(size = c(8, 12)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_resize", list(size = 8), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_resize", list(size = 8), c(2, 3, 21, 16))
  expect_shapes_out_preproc("trafo_pad", list(padding = 2), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_pad", list(padding = c(1, 2)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_pad", list(padding = c(1, 2, 3, 4)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("augment_crop", list(top = 3, left = 4, height = 10, width = 12), c(2, 3, 16, 20))
  # the crop is clamped to the image instead of being padded
  expect_shapes_out_preproc("augment_crop", list(top = 10, left = 10, height = 20, width = 20), c(2, 3, 16, 20))
  # ... while the center crop pads, which torchvision does not do correctly (it swaps height and
  # width), so the extent is only claimed where no padding is needed
  expect_shapes_out_preproc("augment_center_crop", list(size = 8), c(2, 3, 16, 20))
  expect_equal(po("augment_center_crop", size = 32)$shapes_out(list(c(2L, 3L, 16L, 20L)),
    stage = "train")[[1L]], c(2L, 3L, NA, NA))
  expect_shapes_out_preproc("augment_resized_crop", list(top = 1, left = 1, height = 8, width = 8, size = c(4, 4)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_grayscale", list(num_output_channels = 1), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_grayscale", list(num_output_channels = 3), c(2, 3, 16, 20))
  # `transform_rgb_to_grayscale()` drops the channel dimension
  expect_shapes_out_preproc("trafo_rgb_to_grayscale", list(), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_adjust_gamma", list(gamma = 0.5), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_adjust_brightness", list(brightness_factor = 0.5), c(2, 3, 16, 20))
  expect_shapes_out_preproc("augment_color_jitter", list(), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_reshape", list(shape = c(-1, 3, 320)), c(2, 3, 16, 20))
  expect_shapes_out_preproc("trafo_nop", list(), c(2, 3, 16, 20))
})

test_that("no package PipeOp falls back to the traced shape inference", {
  # `infer_shapes()` traces the operator with concrete values, which cannot be exact: it is only
  # for operators that the user supplies (`nn_fn` without `shapes_out`, `pipeop_preproc_torch()`
  # with `shapes_out = "infer"`). Every operator that mlr3torch ships computes its shapes itself.
  traced = Filter(function(key) {
    obj = suppressWarnings(try(po(key), silent = TRUE))
    if (inherits(obj, "try-error") || !inherits(obj, "PipeOpTaskPreprocTorch")) return(FALSE)
    body = paste(deparse(get_private(obj)$.shapes_out), collapse = " ")
    grepl("infer_shapes", body, fixed = TRUE)
  }, mlr_pipeops$keys())
  expect_equal(traced, character(0))
})

test_that("operators reject a 'dim' that does not address a dimension", {
  # assigning to an out-of-range index would extend the shape with `NA`s and build a graph on a
  # shape that no tensor can have
  expect_error(po("nn_glu", dim = 4L)$shapes_out(list(c(NA, 6L, 8L))),
    "cannot use 'dim' 4 for the input shape (NA,6,8), which has 3 dimension(s)", fixed = TRUE)
  expect_error(po("nn_softmax", dim = 9L)$shapes_out(list(c(2L, 4L))), "cannot use 'dim' 9", fixed = TRUE)
  expect_error(po("nn_unsqueeze", dim = 9L)$shapes_out(list(c(2L, 4L))), "cannot use 'dim' 9", fixed = TRUE)
  expect_error(po("nn_squeeze", dim = -4L)$shapes_out(list(c(2L, 1L, 6L))), "cannot use 'dim' -4", fixed = TRUE)
  # negative values are legal in torch and must be accepted
  expect_equal(po("nn_glu", dim = -2L)$shapes_out(list(c(NA, 6L, 8L)))[[1L]], c(NA, 3L, 8L))
  expect_equal(po("nn_glu", dim = -1)$shapes_out(list(c(NA, 6L, 8L)))[[1L]], c(NA, 6L, 4L))
  expect_equal(po("nn_softmax", dim = -1)$shapes_out(list(c(2L, 4L)))[[1L]], c(2L, 4L))
  expect_equal(po("nn_flatten", end_dim = -2L)$shapes_out(list(c(2L, 4L, 6L, 8L)))[[1L]], c(2L, 24L, 8L))
})

test_that("nn_squeeze resolves a vector 'dim' before the module sees it", {
  # the module squeezes from the back, so a negative index would otherwise be resolved against
  # the already shrunk tensor and remove a different dimension
  expect_shapes_out_torch("nn_squeeze", list(dim = c(-1L, -2L)), c(4, 1, 1))
  expect_shapes_out_torch("nn_squeeze", list(dim = c(-3L, 4L)), c(4, 1, 8, 1))
  # a duplicated dimension is harmless
  expect_equal(po("nn_squeeze", dim = c(2L, 2L))$shapes_out(list(c(4L, 1L, 1L, 8L)))[[1L]], c(4L, 1L, 8L))
})

test_that("nn_reshape rejects target dimensions that cannot exist", {
  # such a dimension must not be reported as *known* and propagated into the rest of the graph
  expect_error(po("nn_reshape", shape = c(-5, -7))$shapes_out(list(c(NA, 4L, 6L))),
    "every dimension must be at least 1", fixed = TRUE)
  expect_error(po("nn_reshape", shape = c(0, 24))$shapes_out(list(c(NA, 4L, 6L))),
    "every dimension must be at least 1", fixed = TRUE)
  # the message echoes the shape the user passed, not the internally rewritten one
  expect_error(po("nn_reshape", shape = c(-1, 7))$shapes_out(list(c(2L, 4L, 6L))), "(-1,7)", fixed = TRUE)
})

test_that("convolutions and pooling require the batch dimension and a non-empty output", {
  # dimension 2 is only the channel dimension when the batch dimension is present, otherwise the
  # module is built with a spatial extent as `in_channels`
  expect_error(po("nn_conv2d", out_channels = 4, kernel_size = 3)$shapes_out(list(c(3L, 17L, 19L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  expect_error(po("nn_max_pool2d", kernel_size = 2)$shapes_out(list(c(NA, 28L, 28L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  # a kernel that does not fit gives negative sizes, a stride of 0 gives `Inf`
  expect_error(po("nn_conv2d", out_channels = 4, kernel_size = 9)$shapes_out(list(c(2L, 3L, 4L, 4L))),
    "the output would have the size", fixed = TRUE)
  expect_error(po("nn_max_pool2d", kernel_size = 20, stride = 1)$shapes_out(list(c(NA, 3L, 8L, 8L))),
    "the output would have the size", fixed = TRUE)
  expect_error(po("nn_max_pool1d", kernel_size = 0)$shapes_out(list(c(2L, 3L, 8L))),
    "the output would have the size", fixed = TRUE)
  # `padding` must not partial-match `padding_mode`
  expect_equal(po("nn_conv2d", out_channels = 4, kernel_size = 3, padding_mode = "zeros")$
    shapes_out(list(c(2L, 3L, 8L, 8L)))[[1L]], c(2L, 4L, 6L, 6L))
  # torch supports a per-dimension dilation for max pooling
  expect_equal(po("nn_max_pool2d", kernel_size = 3, dilation = c(1, 2))$
    shapes_out(list(c(2L, 3L, 8L, 8L)))[[1L]], c(2L, 3L, 2L, 2L))
})

test_that("parameter bounds are correct", {
  # 2 is a valid `lambd`, -0.5 is not
  expect_equal(po("nn_softshrink", lambd = 2)$shapes_out(list(c(NA, 3L)))[[1L]], c(NA, 3L))
  expect_error(po("nn_softshrink", lambd = -0.5), "lambd")
})

test_that("shapes that depend on the task come from the task", {
  # the token count is the number of categorical features, not the input dimension
  expect_equal(po("nn_tokenizer_categ", d_token = 5L)$shapes_out(list(c(3L, 1L)),
    task = tsk("breast_cancer"))[[1L]], c(3L, 9L, 5L))
  # the module agrees (it needs an integer tensor, so it is checked directly)
  obj = po("nn_tokenizer_categ", d_token = 5L)
  module = get_private(obj)$.make_module(list(c(3L, 1L)), obj$param_set$get_values(), tsk("breast_cancer"))
  expect_equal(dim(with_no_grad(module(torch_ones(3L, 1L, dtype = torch_long())))), c(3, 9, 5))
  # without a task the number of output features is unknown, as documented
  expect_equal(po("nn_head")$shapes_out(list(c(NA, 16L)))[[1L]], c(NA_integer_, NA_integer_))
})

test_that("nn_block handles a known batch size and an empty block", {
  block = as_graph(po("nn_linear", out_features = 10L)) %>>% po("nn_relu")
  expect_equal(po("nn_block", block, n_blocks = 2L)$shapes_out(list(c(2L, 4L)),
    task = tsk("iris"))[[1L]], c(2L, 10L))
  expect_equal(po("nn_block", block, n_blocks = 0L)$shapes_out(list(c(NA, 4L)),
    task = tsk("iris"))[[1L]], c(NA, 4L))
})

test_that("preprocessing rules follow torchvision, including where it misbehaves", {
  shapes_out = function(id, pv, shape) {
    obj = po(id)
    if (length(pv)) obj$param_set$set_values(.values = pv)
    obj$shapes_out(list(as.integer(shape)), stage = "train")[[1L]]
  }
  # `transform_crop()` clamps to the image and never pads
  expect_equal(shapes_out("augment_crop", list(top = 10, left = 10, height = 20, width = 20),
    c(2, 3, 16, 20)), c(2L, 3L, 7L, 11L))
  expect_equal(shapes_out("augment_crop", list(top = 20, left = 4, height = 5, width = 5),
    c(2, 3, 16, 20)), c(2L, 3L, 0L, 5L))
  # `transform_resized_crop()` crops and then resizes, and a scalar `size` preserves the aspect
  # ratio of the *cropped* image
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
