test_that("PipeOpTorchTransformerEncoderLayer autotest", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_transformer_encoder_layer", nhead = 2)

  expect_pipeop_torch(graph, "nn_transformer_encoder_layer", task)
})

test_that("PipeOpTorchTransformerEncoderLayer paramtest", {
  po_layer = po("nn_transformer_encoder_layer", nhead = 2)
  # d_model is inferred from the input shape, batch_first is fixed to TRUE
  res = expect_paramset(po_layer, nn_transformer_encoder_layer,
    exclude = c("d_model", "batch_first"))
  expect_paramtest(res)
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
  expect_class(module, "nn_transformer_encoder_layer")
  # d_model is the last dimension of the input
  expect_equal(module$linear1$in_features, 4)
  expect_equal(module$linear1$out_features, 16)
  # the input is (batch, sequence, feature), see the "Tensor Layout" section
  expect_true(module$self_attn$batch_first)

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
  expect_true(module$norm_first)
  expect_equal(module$dropout$p, 0.5)
  expect_equal(module$norm1$eps, 1e-3)
  # bias = FALSE also turns off the affine parameters of the layer norms
  expect_null(module$linear1$bias)
  expect_null(module$norm1$weight)
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
