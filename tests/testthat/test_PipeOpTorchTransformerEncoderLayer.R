test_that("PipeOpTorchTransformerEncoderLayer autotest", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_transformer_encoder_layer", nhead = 2)

  expect_pipeop_torch(graph, "nn_transformer_encoder_layer", task, "nn_encoder_layer")
})

test_that("PipeOpTorchTransformerEncoderLayer paramtest", {
  po_layer = po("nn_transformer_encoder_layer", nhead = 2)
  # d_model is inferred from the input shape, batch_first is fixed to TRUE and mask_inputs follows
  # from the construction arguments
  res = expect_paramset(po_layer, nn_encoder_layer,
    exclude = c("d_model", "batch_first", "mask_inputs"))
  expect_paramtest(res)
})

test_that("the construction arguments determine the input channels", {
  po1 = po("nn_transformer_encoder_layer", nhead = 2)
  expect_equal(po1$input$name, "input")
  expect_equal(po1$output$name, "output")

  po2 = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE)
  expect_equal(po2$input$name, c("input", "src_mask"))

  po3 = po("nn_transformer_encoder_layer", nhead = 2, src_key_padding_mask = TRUE)
  expect_equal(po3$input$name, c("input", "src_key_padding_mask"))

  po4 = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE,
    src_key_padding_mask = TRUE)
  expect_equal(po4$input$name, c("input", "src_mask", "src_key_padding_mask"))

  # they are construction arguments and not hyperparameters
  expect_true("src_mask" %nin% po1$param_set$ids())
  expect_true("src_key_padding_mask" %nin% po1$param_set$ids())

  expect_error(po("nn_transformer_encoder_layer", nhead = 2, src_mask = 2), "src_mask")
})

test_that("the construction arguments influence the phash", {
  po1 = po("nn_transformer_encoder_layer", nhead = 2)
  po2 = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE)
  po3 = po("nn_transformer_encoder_layer", nhead = 2, src_key_padding_mask = TRUE)
  po4 = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE,
    src_key_padding_mask = TRUE)

  expect_equal(length(unique(c(po1$phash, po2$phash, po3$phash, po4$phash))), 4L)
  expect_equal(po4$phash,
    po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE,
      src_key_padding_mask = TRUE)$phash)
})

test_that("PipeOpTorchTransformerEncoderLayer shapes_out preserves the input shape", {
  po_layer = po("nn_transformer_encoder_layer", nhead = 2)
  expect_equal(po_layer$shapes_out(list(c(NA, 5, 4))), list(output = c(NA, 5, 4)))
  expect_equal(po_layer$shapes_out(list(c(3, 5, 8))), list(output = c(3, 5, 8)))

  # the sequence length is only needed at runtime, so it may stay unknown
  expect_equal(po_layer$shapes_out(list(c(NA, NA, 4))), list(output = c(NA, NA, 4)))
  expect_equal(po_layer$shapes_out(list(c(3, NA, 4))), list(output = c(3, NA, 4)))
})

test_that("PipeOpTorchTransformerEncoderLayer shapes_out rejects invalid inputs", {
  po_layer = po("nn_transformer_encoder_layer", nhead = 2)

  # the layout is (batch, sequence, feature), so the input must be three-dimensional
  expect_error(po_layer$shapes_out(list(c(NA, 4))), "requires an input with 3 dimensions")
  expect_error(po_layer$shapes_out(list(c(NA, 5, 4, 4))), "requires an input with 3 dimensions")

  # the feature dimension becomes d_model, which sizes the weights, so it has to be known
  expect_error(po_layer$shapes_out(list(c(NA, 5, NA))), "'d_model'")

  # d_model must be divisible by nhead
  expect_error(po_layer$shapes_out(list(c(NA, 5, 5))), "divisible")
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_transformer_encoder_layer",
    list(nhead = 2, dim_feedforward = 8), c(2, 7, 16))
})

test_that("shape inference agrees with the module for random shapes", {
  # gen_shape() draws even dimensions, so the feature dimension stays divisible by nhead
  expect_shape_inference("nn_transformer_encoder_layer",
    list(nhead = 2, dim_feedforward = 8), generators = gen_shape(3L))
})

test_that("PipeOpTorchTransformerEncoderLayer forward works", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_transformer_encoder_layer", nhead = 2)

  md = graph$train(task)[[1L]]
  expect_equal(md$pointer_shape, c(NA, 1, 4))
  net = model_descriptor_to_module(md)
  # the net starts at the ingress, so it is fed (batch, feature) and unsqueezes to (batch, 1, 4)
  out = with_no_grad(net(torch_randn(3, 4)))
  expect_equal(out$shape, c(3, 1, 4))
})

test_that("PipeOpTorchTransformerEncoderLayer infers d_model and fixes batch_first", {
  po_layer = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 16)
  shapes_in = list(input = c(NA, 5, 4))
  module = po_layer$.__enclos_env__$private$.make_module(
    shapes_in, po_layer$param_set$get_values(), NULL
  )
  expect_class(module, "nn_encoder_layer")
  expect_class(module$layer, "nn_transformer_encoder_layer")
  # d_model is the last dimension of the input
  expect_equal(module$layer$linear1$in_features, 4)
  expect_equal(module$layer$linear1$out_features, 16)
  # the input is (batch, sequence, feature), see the "Tensor Layout" section
  expect_true(module$layer$self_attn$batch_first)

  out = with_no_grad(module(torch_randn(3, 5, 4)))
  expect_equal(out$shape, c(3, 5, 4))
  expect_compatible_shapes(po_layer$shapes_out(shapes_in), list(dim(out)))
})

test_that("PipeOpTorchTransformerEncoderLayer is shape-preserving for varying sequence lengths", {
  # the shape inference leaves an unknown sequence length unknown, so the same module must accept
  # different sequence lengths at runtime
  po_layer = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8)
  module = po_layer$.__enclos_env__$private$.make_module(
    list(input = c(NA, NA, 4)), po_layer$param_set$get_values(), NULL
  )
  expect_equal(with_no_grad(module(torch_randn(3, 5, 4)))$shape, c(3, 5, 4))
  expect_equal(with_no_grad(module(torch_randn(3, 9, 4)))$shape, c(3, 9, 4))
})

test_that("PipeOpTorchTransformerEncoderLayer can be stacked", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_transformer_encoder_layer", id = "layer1", nhead = 2, dim_feedforward = 8) %>>%
    po("nn_transformer_encoder_layer", id = "layer2", nhead = 4, dim_feedforward = 8)

  md = graph$train(task)[[1L]]
  expect_equal(md$pointer_shape, c(NA, 1, 4))
  net = model_descriptor_to_module(md)
  out = with_no_grad(net(torch_randn(3, 4)))
  expect_equal(out$shape, c(3, 1, 4))
})

test_that("PipeOpTorchTransformerEncoderLayer parameters are passed to the module", {
  po_layer = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8,
    dropout = 0.5, layer_norm_eps = 1e-3, norm_first = TRUE, bias = FALSE)
  module = po_layer$.__enclos_env__$private$.make_module(
    list(input = c(NA, 5, 4)), po_layer$param_set$get_values(), NULL
  )
  expect_true(module$layer$norm_first)
  expect_equal(module$layer$dropout$p, 0.5)
  expect_equal(module$layer$norm1$eps, 1e-3)
  # bias = FALSE also turns off the affine parameters of the layer norms
  expect_null(module$layer$linear1$bias)
  expect_null(module$layer$norm1$weight)
})

test_that("PipeOpTorchTransformerEncoderLayer activation", {
  po_relu = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8)
  po_gelu = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8,
    activation = "gelu")
  po_fn = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8,
    activation = function(x) torch_sigmoid(x))

  make = function(po_layer) {
    po_layer$.__enclos_env__$private$.make_module(
      list(input = c(NA, 5, 4)), po_layer$param_set$get_values(), NULL
    )
  }
  x = torch_randn(2, 5, 4)
  walk(list(po_relu, po_gelu, po_fn), function(po_layer) {
    expect_equal(with_no_grad(make(po_layer)(x))$shape, c(2, 5, 4))
  })

  # only "relu", "gelu" and functions are accepted
  expect_error(po("nn_transformer_encoder_layer", nhead = 2, activation = "bogus"), "activation")
})

test_that("the mask channels leave the output shape alone and are checked", {
  po_both = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE,
    src_key_padding_mask = TRUE)
  # the masks say which positions are attended to, so the output is still the input shape
  expect_equal(
    po_both$shapes_out(list(c(NA, 5, 4), c(5, 5), c(NA, 5))),
    list(output = c(NA, 5, 4))
  )
  # a src_mask may also be given per batch element and head
  expect_equal(
    po_both$shapes_out(list(c(NA, 5, 4), c(6, 5, 5), c(NA, 5))),
    list(output = c(NA, 5, 4))
  )
  # the sizes of the masks may be unknown
  expect_equal(
    po_both$shapes_out(list(c(NA, NA, 4), c(NA, NA), c(NA, NA))),
    list(output = c(NA, NA, 4))
  )

  po_mask = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE)
  expect_error(po_mask$shapes_out(list(c(NA, 5, 4), c(2, 5, 5, 5))),
    "'src_mask' expects a shape with 2 or 3 dimensions")

  po_pad = po("nn_transformer_encoder_layer", nhead = 2, src_key_padding_mask = TRUE)
  expect_error(po_pad$shapes_out(list(c(NA, 5, 4), c(2, 5, 5))),
    "'src_key_padding_mask' expects a shape with 2 dimensions")

  # the number of shapes must match the number of channels
  expect_error(po_pad$shapes_out(list(c(NA, 5, 4))), "input channel")
})

test_that("is_causal cannot be combined with the src_mask channel", {
  po_both = po("nn_transformer_encoder_layer", nhead = 2, src_mask = TRUE, is_causal = TRUE)
  expect_error(po_both$shapes_out(list(c(NA, 5, 4), c(5, 5))),
    "'is_causal' cannot be combined with the 'src_mask' input channel")

  # without the channel it is fine
  po_causal = po("nn_transformer_encoder_layer", nhead = 2, is_causal = TRUE)
  expect_equal(po_causal$shapes_out(list(c(NA, 5, 4))), list(output = c(NA, 5, 4)))
})

test_that("is_causal makes the layer attend only to the past", {
  po_causal = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8, dropout = 0,
    is_causal = TRUE)
  module = po_causal$.__enclos_env__$private$.make_module(
    list(input = c(NA, 5, 4)), po_causal$param_set$get_values(), NULL
  )
  module$eval()

  x = torch_randn(2, 5, 4)
  out = with_no_grad(module(x))
  expect_equal(out$shape, c(2, 5, 4))

  # changing the last position must not change the outputs of the earlier ones
  x2 = x$clone()
  x2[, 5, ] = torch_randn(2, 4)
  out2 = with_no_grad(module(x2))
  expect_true(torch_allclose(out[, 1:4, ], out2[, 1:4, ], atol = 1e-5))
  expect_false(torch_allclose(out[, 5, ], out2[, 5, ], atol = 1e-5))

  # without is_causal the earlier positions do see the change
  po_full = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8, dropout = 0)
  module_full = po_full$.__enclos_env__$private$.make_module(
    list(input = c(NA, 5, 4)), po_full$param_set$get_values(), NULL
  )
  module_full$eval()
  expect_false(torch_allclose(
    with_no_grad(module_full(x))[, 1:4, ], with_no_grad(module_full(x2))[, 1:4, ], atol = 1e-5))
})

test_that("the mask inputs reach the wrapped module", {
  po_both = po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8, dropout = 0,
    src_mask = TRUE, src_key_padding_mask = TRUE)
  module = po_both$.__enclos_env__$private$.make_module(
    list(input = c(NA, 5, 4), src_mask = c(5, 5), src_key_padding_mask = c(NA, 5)),
    po_both$param_set$get_values(), NULL
  )
  module$eval()
  expect_equal(module$mask_inputs, c("src_mask", "src_key_padding_mask"))

  x = torch_randn(2, 5, 4)
  src_mask = torch_zeros(5, 5, dtype = torch_bool())
  src_mask[1, 5] = TRUE
  padding = torch_zeros(2, 5, dtype = torch_bool())
  padding[1, 5] = TRUE

  # the inputs arrive in the order of the input channels and are routed to the right argument
  observed = with_no_grad(module(x, src_mask, padding))
  expected = with_no_grad(module$layer(src = x, src_mask = src_mask, src_key_padding_mask = padding))
  expect_true(torch_allclose(observed, expected))

  # the masks make a difference
  unmasked = with_no_grad(module$layer(src = x))
  expect_false(torch_allclose(observed, unmasked, atol = 1e-5))
})

test_that("the mask channels work inside a graph", {
  task = tsk("iris")
  # the padding mask is built from the features so that it flows through the graph as a tensor
  graph = po("torch_ingress_num") %>>%
    list(
      po("nn_unsqueeze", dim = 2),
      po("nn_fn", id = "padding", fn = function(x) (x < -1e9)[, 1:1])
    ) %>>%
    po("nn_transformer_encoder_layer", nhead = 2, dim_feedforward = 8, dropout = 0,
      src_key_padding_mask = TRUE)

  md = graph$train(task)[[1L]]
  expect_equal(md$pointer_shape, c(NA, 1, 4))
  net = model_descriptor_to_module(md)
  out = with_no_grad(net(torch_randn(3, 4)))
  expect_equal(out$shape, c(3, 1, 4))
})
