test_that("PipeOpTorchMultiheadAttention works for self-attention", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_multihead_attention", num_heads = 2, batch_first = TRUE)

  expect_pipeop_torch(graph, "nn_multihead_attention", task, "nn_attention")
})

test_that("PipeOpTorchMultiheadAttention paramtest", {
  po_attention = po("nn_multihead_attention", num_heads = 2)
  # embed_dim, kdim and vdim are inferred from the input shapes, need_weights is a construction arg
  res = expect_paramset(po_attention, nn_attention,
    exclude = c("embed_dim", "kdim", "vdim", "need_weights"))
  expect_paramtest(res)
})

test_that("PipeOpTorchMultiheadAttention mode determines the input channels", {
  po1 = po("nn_multihead_attention", num_heads = 2)
  expect_equal(po1$input$name, "input")

  po2 = po("nn_multihead_attention", mode = "cross", num_heads = 2)
  expect_equal(po2$input$name, c("query", "key_value"))

  po3 = po("nn_multihead_attention", mode = "general", num_heads = 2)
  expect_equal(po3$input$name, c("query", "key", "value"))

  # mode is a construction argument and not a hyperparameter
  expect_true("mode" %nin% po1$param_set$ids())

  expect_error(po("nn_multihead_attention", mode = "bogus"), "mode")
})

test_that("PipeOpTorchMultiheadAttention need_weights determines the output channels", {
  po1 = po("nn_multihead_attention", num_heads = 2)
  expect_equal(po1$output$name, "output")

  po2 = po("nn_multihead_attention", need_weights = TRUE, num_heads = 2)
  expect_equal(po2$output$name, c("output", "weights"))

  # need_weights is a construction argument and not a hyperparameter
  expect_true("need_weights" %nin% po1$param_set$ids())

  expect_error(po("nn_multihead_attention", need_weights = 2), "need_weights")
})

test_that("PipeOpTorchMultiheadAttention mode and need_weights influence the phash", {
  po1 = po("nn_multihead_attention", num_heads = 2)
  po2 = po("nn_multihead_attention", mode = "cross", num_heads = 2)
  po3 = po("nn_multihead_attention", mode = "general", num_heads = 2)
  po4 = po("nn_multihead_attention", need_weights = TRUE, num_heads = 2)
  po5 = po("nn_multihead_attention", mode = "cross", need_weights = TRUE, num_heads = 2)

  hashes = c(po1$phash, po2$phash, po3$phash, po4$phash, po5$phash)
  expect_equal(length(unique(hashes)), 5L)

  expect_equal(po1$phash, po("nn_multihead_attention", num_heads = 2)$phash)
  expect_equal(po5$phash,
    po("nn_multihead_attention", mode = "cross", need_weights = TRUE, num_heads = 2)$phash)
})

test_that("PipeOpTorchMultiheadAttention shapes_out for the output channel", {
  # batch-first: (batch, sequence, feature)
  po1 = po("nn_multihead_attention", num_heads = 2, batch_first = TRUE)
  expect_equal(po1$shapes_out(list(c(NA, 5, 4))), list(output = c(NA, 5, 4)))

  # sequence-first: (sequence, batch, feature) -- the output still has the query shape
  po1b = po("nn_multihead_attention", num_heads = 2, batch_first = FALSE)
  expect_equal(po1b$shapes_out(list(c(5, NA, 4))), list(output = c(5, NA, 4)))

  # the output has the shape of the query, not of the key/value input
  po2 = po("nn_multihead_attention", innum = 2, num_heads = 2, batch_first = TRUE)
  expect_equal(po2$shapes_out(list(c(NA, 5, 4), c(NA, 7, 6))), list(output = c(NA, 5, 4)))

  po3 = po("nn_multihead_attention", innum = 3, num_heads = 2, batch_first = TRUE)
  expect_equal(
    po3$shapes_out(list(c(NA, 5, 4), c(NA, 7, 6), c(NA, 7, 8))),
    list(output = c(NA, 5, 4))
  )

  # embed_dim must be divisible by num_heads
  expect_error(po1$shapes_out(list(c(NA, 5, 5))), "divisible")

  # only three-dimensional inputs are allowed
  expect_error(po1$shapes_out(list(c(NA, 4))), "three-dimensional")

  # the feature dimension must be known
  expect_error(po1$shapes_out(list(c(NA, 5, NA))), "last dimension is unknown")
})

test_that("PipeOpTorchMultiheadAttention shapes_out for the weights channel", {
  # The attention weights are ALWAYS batch-first, irrespective of the `batch_first` parameter.
  # Averaged over the heads (the default), they are (batch, query_sequence, key_sequence).
  po_bf = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE)
  expect_equal(
    po_bf$shapes_out(list(c(NA, 5, 4))),
    list(output = c(NA, 5, 4), weights = c(NA, 5, 5))
  )

  po_sf = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = FALSE)
  expect_equal(
    po_sf$shapes_out(list(c(5, NA, 4))),
    # output keeps the sequence-first layout, weights are batch-first
    list(output = c(5, NA, 4), weights = c(NA, 5, 5))
  )

  # not averaged: (batch, num_heads, query_sequence, key_sequence)
  po_noavg = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE,
    avg_weights = FALSE)
  expect_equal(
    po_noavg$shapes_out(list(c(NA, 5, 4))),
    list(output = c(NA, 5, 4), weights = c(NA, 2, 5, 5))
  )

  # cross-attention: the key sequence length comes from the key input
  po_cross = po("nn_multihead_attention", innum = 2, outnum = 2, num_heads = 2, batch_first = TRUE)
  expect_equal(
    po_cross$shapes_out(list(c(NA, 5, 4), c(NA, 7, 4))),
    list(output = c(NA, 5, 4), weights = c(NA, 5, 7))
  )

  # add_bias_kv and add_zero_attn each add one to the key sequence length
  po_bias = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE,
    add_bias_kv = TRUE)
  expect_equal(po_bias$shapes_out(list(c(NA, 5, 4)))$weights, c(NA, 5, 6))

  po_zero = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE,
    add_zero_attn = TRUE)
  expect_equal(po_zero$shapes_out(list(c(NA, 5, 4)))$weights, c(NA, 5, 6))

  po_both = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE,
    add_bias_kv = TRUE, add_zero_attn = TRUE)
  expect_equal(po_both$shapes_out(list(c(NA, 5, 4)))$weights, c(NA, 5, 7))
})

test_that("PipeOpTorchMultiheadAttention accounts for torch ignoring avg_weights", {
  # torch::nn_multihead_attention() only forwards `avg_weights` to
  # nnf_multi_head_attention_forward() when kdim == vdim == embed_dim. Otherwise the weights are
  # averaged regardless, so the predicted shape must be the averaged (3d) one.
  po_diff = po("nn_multihead_attention", innum = 2, outnum = 2, num_heads = 2, batch_first = TRUE,
    avg_weights = FALSE)
  expect_equal(
    po_diff$shapes_out(list(c(NA, 5, 4), c(NA, 7, 6)))$weights,
    c(NA, 5, 7)
  )

  # but when the embedding dims agree, avg_weights = FALSE gives the 4d shape
  po_same = po("nn_multihead_attention", innum = 2, outnum = 2, num_heads = 2, batch_first = TRUE,
    avg_weights = FALSE)
  expect_equal(
    po_same$shapes_out(list(c(NA, 5, 4), c(NA, 7, 4)))$weights,
    c(NA, 2, 5, 7)
  )

  # the predicted shapes must match what torch actually produces in both cases
  module_diff = po_diff$.__enclos_env__$private$.make_module(
    list(query = c(NA, 5, 4), key_value = c(NA, 7, 6)), po_diff$param_set$get_values(), NULL
  )
  out_diff = with_no_grad(module_diff(torch_randn(3, 5, 4), torch_randn(3, 7, 6)))
  expect_equal(out_diff$weights$shape, c(3, 5, 7))

  module_same = po_same$.__enclos_env__$private$.make_module(
    list(query = c(NA, 5, 4), key_value = c(NA, 7, 4)), po_same$param_set$get_values(), NULL
  )
  out_same = with_no_grad(module_same(torch_randn(3, 5, 4), torch_randn(3, 7, 4)))
  expect_equal(out_same$weights$shape, c(3, 2, 5, 7))
})

test_that("PipeOpTorchMultiheadAttention self-attention forward works", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_multihead_attention", num_heads = 2, batch_first = TRUE)

  md = graph$train(task)[[1L]]
  expect_equal(md$pointer_shape, c(NA, 1, 4))
  net = model_descriptor_to_module(md)
  out = with_no_grad(net(torch_randn(3, 1, 4)))
  expect_equal(out$shape, c(3, 1, 4))
})

test_that("PipeOpTorchMultiheadAttention with outnum = 2 returns output and weights", {
  po_attention = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE)
  shapes_in = list(input = c(NA, 5, 4))
  module = po_attention$.__enclos_env__$private$.make_module(
    shapes_in, po_attention$param_set$get_values(), NULL
  )
  out = with_no_grad(module(torch_randn(3, 5, 4)))
  expect_list(out, len = 2L)
  expect_equal(names(out), c("output", "weights"))
  expect_equal(out$output$shape, c(3, 5, 4))
  expect_equal(out$weights$shape, c(3, 5, 5))

  expect_compatible_shapes(
    po_attention$shapes_out(shapes_in), list(dim(out$output), dim(out$weights))
  )
})

test_that("PipeOpTorchMultiheadAttention with outnum = 2 works inside a graph", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE)

  mds = graph$train(task)
  expect_equal(length(mds), 2L)
  expect_equal(mds[[1L]]$pointer_shape, c(NA, 1, 4))
  expect_equal(mds[[2L]]$pointer_shape, c(NA, 1, 1))
  expect_equal(mds[[1L]]$pointer[[2L]], "output")
  expect_equal(mds[[2L]]$pointer[[2L]], "weights")
})

test_that("PipeOpTorchMultiheadAttention with outnum = 2 and avg_weights = FALSE", {
  po_attention = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = TRUE,
    avg_weights = FALSE)
  shapes_in = list(input = c(NA, 5, 4))
  module = po_attention$.__enclos_env__$private$.make_module(
    shapes_in, po_attention$param_set$get_values(), NULL
  )
  out = with_no_grad(module(torch_randn(3, 5, 4)))
  expect_equal(out$weights$shape, c(3, 2, 5, 5))
  expect_compatible_shapes(
    po_attention$shapes_out(shapes_in), list(dim(out$output), dim(out$weights))
  )
})

test_that("PipeOpTorchMultiheadAttention weights are batch-first even if batch_first is FALSE", {
  po_attention = po("nn_multihead_attention", outnum = 2, num_heads = 2, batch_first = FALSE)
  shapes_in = list(input = c(5, NA, 4))
  module = po_attention$.__enclos_env__$private$.make_module(
    shapes_in, po_attention$param_set$get_values(), NULL
  )
  # (sequence, batch, feature)
  out = with_no_grad(module(torch_randn(5, 3, 4)))
  expect_equal(out$output$shape, c(5, 3, 4))
  # weights stay (batch, query_sequence, key_sequence)
  expect_equal(out$weights$shape, c(3, 5, 5))
  expect_compatible_shapes(
    po_attention$shapes_out(shapes_in), list(dim(out$output), dim(out$weights))
  )
})

test_that("nn_attention batch_first layouts agree after transposing", {
  torch_manual_seed(1)
  x = torch_randn(3, 5, 4)

  module_bf = nn_attention(embed_dim = 4, num_heads = 2, batch_first = TRUE)
  module_sf = nn_attention(embed_dim = 4, num_heads = 2, batch_first = FALSE)
  module_sf$load_state_dict(module_bf$state_dict())

  module_bf$eval()
  module_sf$eval()
  out_bf = with_no_grad(module_bf(x))
  out_sf = with_no_grad(module_sf(x$transpose(1, 2)))

  expect_true(torch_allclose(out_bf, out_sf$transpose(1, 2), atol = 1e-6))
})

test_that("PipeOpTorchMultiheadAttention cross-attention with two inputs works", {
  task = tsk("iris")
  task_query = task$clone()$select(c("Sepal.Length", "Sepal.Width"))
  task_context = task$clone()$select(c("Petal.Length", "Petal.Width"))

  graph_query = po("torch_ingress_num_1") %>>%
    po("nn_unsqueeze", id = "unsqueeze_query", dim = 2)
  graph_context = po("torch_ingress_num_2") %>>%
    po("nn_unsqueeze", id = "unsqueeze_context", dim = 2)

  graph = gunion(list(graph_query, graph_context)) %>>%
    po("nn_multihead_attention", innum = 2, num_heads = 2, batch_first = TRUE)

  md = graph$train(list(task_query, task_context), single_input = FALSE)[[1L]]
  # output shape is the query shape
  expect_equal(md$pointer_shape, c(NA, 1, 2))

  net = model_descriptor_to_module(md)
  out = with_no_grad(net(torch_randn(3, 1, 2), torch_randn(3, 1, 2)))
  expect_equal(out$shape, c(3, 1, 2))

  # kdim / vdim are inferred, so query and key/value may have different embedding sizes
  po_attention = po("nn_multihead_attention", innum = 2, num_heads = 2, batch_first = TRUE)
  module = po_attention$.__enclos_env__$private$.make_module(
    list(query = c(NA, 5, 4), key_value = c(NA, 7, 6)),
    po_attention$param_set$get_values(),
    NULL
  )
  out = with_no_grad(module(torch_randn(3, 5, 4), torch_randn(3, 7, 6)))
  expect_equal(out$shape, c(3, 5, 4))
})

test_that("PipeOpTorchMultiheadAttention cross-attention with three inputs works", {
  po_attention = po("nn_multihead_attention", innum = 3, num_heads = 2, batch_first = TRUE)
  module = po_attention$.__enclos_env__$private$.make_module(
    list(query = c(NA, 5, 4), key = c(NA, 7, 6), value = c(NA, 7, 8)),
    po_attention$param_set$get_values(),
    NULL
  )
  out = with_no_grad(module(torch_randn(3, 5, 4), torch_randn(3, 7, 6), torch_randn(3, 7, 8)))
  expect_equal(out$shape, c(3, 5, 4))
})

test_that("nn_attention with one input is equivalent to torch self-attention", {
  torch_manual_seed(1)
  module = nn_attention(embed_dim = 4, num_heads = 2, batch_first = TRUE)
  module$eval()
  x = torch_randn(3, 5, 4)
  expected = with_no_grad(
    module$attention(query = x, key = x, value = x, need_weights = FALSE)[[1L]]
  )
  observed = with_no_grad(module(x))
  expect_true(torch_allclose(expected, observed))
})
