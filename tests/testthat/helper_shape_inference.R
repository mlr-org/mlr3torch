# Helpers that verify shape inference against what the operator actually does.
#
# The ground truth is obtained by building the module from the (possibly partially unknown) shape
# and running a tensor through it -- on torch's "meta" device where possible, which computes
# shapes without allocating memory, falling back to the cpu for operators torch does not implement
# there. Note that the meta device is *not* reliable for torchvision's advanced indexing, so the
# preprocessing operators are always run on the cpu.

run_on_shape = function(fn, shape, n_in = 1L, device = c("meta", "cpu")) {
  make = function(device) {
    tensor = mlr3misc::invoke(torch_empty, .args = as.list(as.integer(shape)),
      device = torch_device(device))
    rep(list(tensor), n_in)
  }
  out = if (identical(device[1L], "cpu")) {
    with_no_grad(mlr3misc::invoke(fn, .args = make("cpu")))
  } else {
    tryCatch(
      with_no_grad(mlr3misc::invoke(fn, .args = make("meta"))),
      error = function(e) with_no_grad(mlr3misc::invoke(fn, .args = make("cpu")))
    )
  }
  dim(out)
}

# Ground truth for a `PipeOpTorch`. The module is built from `shape_in` -- the possibly partially
# unknown shape that the inference saw -- and is then run on a tensor of the concrete `shape`,
# which is exactly what happens when a network is trained: the module is constructed from the
# shapes of the graph and afterwards sees real data.
true_shape_torch = function(obj, shape_in, shape, n_in = 1L, task = NULL) {
  shapes_in = rep(list(as.integer(shape_in)), n_in)
  # `$train()` names the shapes after the input channels, so `.make_module()` may rely on it
  if ("..." %nin% obj$input$name) names(shapes_in) = obj$input$name
  module = get_private(obj)$.make_module(shapes_in, obj$param_set$get_values(), task)
  run_on_shape(module, shape, n_in)
}

# ground truth for a `PipeOpTaskPreprocTorch`: apply its function, respecting `rowwise`
true_shape_preproc = function(obj, shape) {
  pv = obj$param_set$get_values(tags = "train")
  pv$stages = NULL
  pv$affect_columns = NULL
  fn = obj$fn
  args = pv[intersect(names(pv), names(formals(fn)))]
  f = function(x) mlr3misc::invoke(fn, x, .args = args)
  if (obj$rowwise) {
    c(as.integer(shape[1L]), run_on_shape(f, shape[-1L], device = "cpu"))
  } else {
    run_on_shape(f, shape, device = "cpu")
  }
}

make_po = function(id, param_vals) {
  obj = po(id)
  if (length(param_vals)) obj$param_set$set_values(.values = param_vals)
  obj
}

# parameter values may be functions (`nn_module_generator`s), so only their names are shown
make_label = function(id, param_vals) {
  sprintf("%s(%s)", id, paste(names(param_vals), collapse = ", "))
}

# For a fully known shape, the inferred shape must be exactly what the operator returns.
# Afterwards every dimension is set to `NA` in turn (and the batch dimension together with each
# other one, which is the transformer case): the inference may reject such a shape -- that is what
# `assert_known_dims()` does -- but if it returns a shape, every dimension it claims to know must
# still agree with the ground truth, and the number of dimensions must match.
expect_shape_inference = function(shape, inferred_shape, true_shape, label) {
  shape = as.integer(shape)
  expect_equal(inferred_shape(shape), true_shape(shape, shape),
    label = sprintf("%s: known shape %s", label, shape_to_str(shape)))

  na_idx = c(as.list(seq_along(shape)), lapply(seq_along(shape)[-1L], function(i) c(1L, i)))
  for (idx in na_idx) {
    partial = shape
    partial[idx] = NA_integer_
    inferred = tryCatch(inferred_shape(partial), error = function(e) e)
    if (inherits(inferred, "condition")) {
      # a rejection must be a readable R error, not a C++ error from libtorch
      expect_false(grepl("SymInt|IntArrayRef|Exception raised|INTERNAL ASSERT",
        conditionMessage(inferred)),
        label = sprintf("%s: readable error for %s", label, shape_to_str(partial)))
      next
    }
    truth = true_shape(partial, shape)
    expect_equal(length(inferred), length(truth),
      label = sprintf("%s: number of dimensions for %s", label, shape_to_str(partial)))
    known = !is.na(inferred)
    expect_equal(inferred[known], truth[known],
      label = sprintf("%s: known dimensions for %s", label, shape_to_str(partial)))
  }
}

expect_shapes_out_torch = function(id, param_vals, shape, n_in = 1L, task = NULL) {
  obj = make_po(id, param_vals)
  expect_shape_inference(shape,
    inferred_shape = function(s) obj$shapes_out(rep(list(s), n_in), task = task)[[1L]],
    true_shape = function(shape_in, s) true_shape_torch(obj, shape_in, s, n_in = n_in, task = task),
    label = make_label(id, param_vals)
  )
}

expect_shapes_out_preproc = function(id, param_vals, shape) {
  obj = make_po(id, param_vals)
  expect_shape_inference(shape,
    inferred_shape = function(s) obj$shapes_out(list(s), stage = "train")[[1L]],
    # the preprocessing function does not depend on the input shape
    true_shape = function(shape_in, s) true_shape_preproc(obj, s),
    label = make_label(id, param_vals)
  )
}

# --- randomized checking -------------------------------------------------------------------
#
# How many (parameter, shape) draws are checked per PipeOp. Every draw additionally sweeps all
# single-`NA` patterns and the batch-plus-one patterns, so a budget of 3 already means dozens of
# comparisons per operator. Raise it while working on the shape inference:
#     MLR3TORCH_SHAPE_BUDGET=50 Rscript -e 'testthat::test_local()'
shape_inference_budget = function(default = 3L) {
  budget = Sys.getenv("MLR3TORCH_SHAPE_BUDGET")
  if (!nzchar(budget)) {
    return(default)
  }
  assert_int(as.integer(budget), lower = 1L)
}

# samplers for the parameters and the input rank of a PipeOp; `size()` draws a dimension size
size = function(n = 1L, from = 4L, to = 12L) sample(seq(from, to), n, replace = TRUE)

# The specification of one operator: `rank` is the number of input dimensions (the first one is
# always the batch dimension), `params` draws a set of parameter values, and `n_in` is the number
# of input channels. Everything not listed here is shape-preserving and needs no parameters.
shape_inference_specs = function() {
  conv = function(d) list(rank = d + 2L, params = function() {
    list(out_channels = sample(1:4, 1L), kernel_size = sample(1:3, 1L), stride = sample(1:2, 1L),
      padding = sample(0:1, 1L), dilation = sample(1:2, 1L))
  })
  conv_transpose = function(d) list(rank = d + 2L, params = function() {
    list(out_channels = sample(1:4, 1L), kernel_size = sample(1:3, 1L), stride = sample(1:2, 1L),
      padding = sample(0:1, 1L))
  })
  pool = function(d, dilation = FALSE) list(rank = d + 2L, params = function() {
    pv = list(kernel_size = sample(1:3, 1L), stride = sample(1:2, 1L), padding = 0L,
      ceil_mode = sample(c(TRUE, FALSE), 1L))
    if (dilation) pv$dilation = sample(1:2, 1L)
    pv
  })
  adaptive = function(d) list(rank = d + 2L, params = function() list(output_size = sample(1:4, d)))

  c(
    list(
      nn_linear = list(rank = 3L, params = function() list(out_features = sample(1:8, 1L))),
      nn_head = list(rank = 2L, params = function() list(), task = tsk("iris")),
      nn_layer_norm = list(rank = 4L, params = function() list(dims = sample(1:3, 1L))),
      nn_ft_cls = list(rank = 3L, params = function() list(initialization = "uniform")),
      nn_tokenizer_num = list(rank = 2L, params = function() list(d_token = sample(2:6, 1L))),
      nn_batch_norm1d = list(rank = 3L, params = function() list()),
      nn_batch_norm2d = list(rank = 4L, params = function() list()),
      nn_batch_norm3d = list(rank = 5L, params = function() list()),
      nn_flatten = list(rank = 4L, params = function() list(start_dim = 2L, end_dim = sample(2:3, 1L))),
      nn_unsqueeze = list(rank = 3L, params = function() list(dim = sample(c(1:4, -1L), 1L))),
      nn_squeeze = list(rank = 3L, params = function() list(dim = sample(2:3, 1L)), even = FALSE),
      nn_softmax = list(rank = 3L, params = function() list(dim = sample(c(2:3, -1L), 1L))),
      nn_dropout = list(rank = 3L, params = function() list(p = 0.5)),
      nn_glu = list(rank = 3L, params = function() list(dim = sample(c(2:3, -1L), 1L))),
      nn_merge_sum = list(rank = 3L, params = function() list(), n_in = 2L),
      nn_merge_prod = list(rank = 3L, params = function() list(), n_in = 2L),
      nn_merge_cat = list(rank = 3L, params = function() list(dim = sample(2:3, 1L)), n_in = 2L),
      nn_prelu = list(rank = 3L, params = function() list(num_parameters = 1L)),
      nn_rrelu = list(rank = 3L, params = function() list(lower = 0.1, upper = 0.3)),
      nn_threshold = list(rank = 3L, params = function() list(threshold = 0.5, value = 0)),
      # the target must match the number of elements per observation, so the shape is not doubled
      nn_reshape = list(rank = 3L, params = function() list(shape = c(-1L, 24L)),
        fixed_shape = c(3L, 4L, 6L), even = FALSE),
      nn_ft_transformer_block = list(rank = 3L, params = function() {
        # `d_token` is read from the last dimension and must be divisible by the number of heads
        list(attention_n_heads = 2L, ffn_d_hidden = 8L, is_first_layer = TRUE, query_idx = NULL)
      })
    ),
    set_names(lapply(1:3, conv), sprintf("nn_conv%id", 1:3)),
    set_names(lapply(1:3, conv_transpose), sprintf("nn_conv_transpose%id", 1:3)),
    set_names(lapply(1:3, pool), sprintf("nn_avg_pool%id", 1:3)),
    set_names(lapply(1:3, function(d) pool(d, dilation = TRUE)), sprintf("nn_max_pool%id", 1:3)),
    set_names(lapply(1:3, adaptive), sprintf("nn_adaptive_avg_pool%id", 1:3))
  )
}

# the preprocessing operators all work on (batch, channels, height, width)
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
      height = size(1L, 2L, 14L), width = size(1L, 2L, 14L)),
    augment_center_crop = function() list(size = size(1L, 2L, 6L)),
    augment_resized_crop = function() list(top = 1L, left = 1L, height = size(1L, 2L, 8L),
      width = size(1L, 2L, 8L), size = if (sample(c(TRUE, FALSE), 1L)) size(1L, 2L, 6L) else size(2L, 2L, 6L))
  )
}

# Draws `budget` (parameter, shape) combinations for one PipeOpTorch and checks each of them.
# `seed` makes a failure reproducible: it is derived from the operator id and the draw.
expect_shape_inference_sampled = function(id, spec, budget = shape_inference_budget()) {
  rank = spec$rank %??% 3L
  n_in = spec$n_in %??% 1L
  for (i in seq_len(budget)) {
    set.seed(sum(utf8ToInt(id)) + i)
    shape = spec$fixed_shape %??% c(sample(1:3, 1L), size(rank - 1L))
    # the gated units halve a dimension and reshaping needs a divisible number of elements
    if (!identical(spec$even, FALSE)) shape[-1L] = shape[-1L] * 2L
    param_vals = spec$params()
    expect_shapes_out_torch(id, param_vals, shape, n_in = n_in, task = spec$task)
  }
}

expect_shape_inference_sampled_preproc = function(id, params, budget = shape_inference_budget()) {
  for (i in seq_len(budget)) {
    set.seed(sum(utf8ToInt(id)) + i)
    shape = c(sample(1:3, 1L), 3L, size(2L, 6L, 16L))
    expect_shapes_out_preproc(id, params(), shape)
  }
}
