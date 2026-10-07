seq_graph = function(po_test) {
  po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 2) %>>% po_test
}

test_that("PipeOpTorchRNN autotest", {
  expect_pipeop_torch(seq_graph(po("nn_rnn", hidden_size = 5)), "nn_rnn", tsk("iris"),
    "nn_recurrent")
})

test_that("PipeOpTorchLSTM autotest", {
  expect_pipeop_torch(seq_graph(po("nn_lstm", hidden_size = 5)), "nn_lstm", tsk("iris"),
    "nn_recurrent")
})

test_that("PipeOpTorchGRU autotest", {
  expect_pipeop_torch(seq_graph(po("nn_gru", hidden_size = 5)), "nn_gru", tsk("iris"),
    "nn_recurrent")
})

test_that("recurrent layers paramtest", {
  # input_size is inferred from the input shape and batch_first is fixed to TRUE
  excluded = c("input_size", "batch_first")
  expect_paramtest(expect_paramset(po("nn_lstm"), nn_lstm, exclude = excluded))
  expect_paramtest(expect_paramset(po("nn_gru"), nn_gru, exclude = excluded))
  # torch leaves `nonlinearity` at NULL and resolves it to "tanh" itself, which is the default the
  # parameter documents
  expect_paramtest(expect_paramset(po("nn_rnn"), nn_rnn, exclude = excluded,
    exclude_defaults = "nonlinearity"))
})

# all outputs of the module of a recurrent layer, of which only "output" is an output channel by default
all_outputs = function(id) {
  # only an LSTM carries a cell state
  if (id == "nn_lstm") c("output", "h_n", "c_n") else c("output", "h_n")
}

test_that("the final states are only output channels when requested", {
  for (id in c("nn_rnn", "nn_lstm", "nn_gru")) {
    expect_equal(po(id, hidden_size = 5)$output$name, "output")
    expect_equal(po(id, hidden_size = 5, outputs = all_outputs(id))$output$name, all_outputs(id))
  }

  # return_state is set according to `$outputs` and not a hyperparameter
  expect_true("return_state" %nin% po("nn_lstm")$param_set$ids())
  expect_false(po("nn_lstm", hidden_size = 5)$phash ==
    po("nn_lstm", hidden_size = 5, outputs = all_outputs("nn_lstm"))$phash)
  # the three layers are different operators even with the same parameters
  expect_false(po("nn_lstm", hidden_size = 5)$phash == po("nn_gru", hidden_size = 5)$phash)
})

test_that("shape inference matches the operator", {
  for (id in c("nn_rnn", "nn_lstm", "nn_gru")) {
    expect_shape_inference(id, list(hidden_size = 5), c(2, 7, 4))
    expect_shape_inference(id, list(hidden_size = 5, bidirectional = TRUE), c(2, 7, 4))
    expect_shape_inference(id, list(hidden_size = 5, num_layers = 2), c(2, 7, 4))
  }
})

test_that("shape inference matches the operator when the state is returned", {
  # the state is transposed to be batch-first, which the comparison against the module checks
  for (id in c("nn_rnn", "nn_lstm", "nn_gru")) {
    expect_shape_inference(id, list(hidden_size = 5, outputs = all_outputs(id)), c(2, 7, 4))
    expect_shape_inference(id,
      list(hidden_size = 5, outputs = all_outputs(id), num_layers = 2, bidirectional = TRUE),
      c(2, 7, 4))
  }
})

test_that("shape inference needs a sequence with a known feature dimension", {
  expect_error(po("nn_lstm", hidden_size = 5)$shapes_out(list(c(NA, 4L))),
    "requires an input with 3 dimensions", fixed = TRUE)
  expect_error(po("nn_lstm", hidden_size = 5)$shapes_out(list(c(NA, 7L, NA))),
    "'input_size'", fixed = TRUE)
  # the sequence length is only needed at runtime and may stay unknown
  expect_equal(po("nn_lstm", hidden_size = 5)$shapes_out(list(c(NA, NA, 4L)))[[1L]], c(NA, NA, 5L))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  for (id in c("nn_rnn", "nn_lstm", "nn_gru")) {
    expect_shape_inference(id,
      params = function() {
        list(hidden_size = sample(2:6, 1L), bidirectional = sample(c(TRUE, FALSE), 1L),
          num_layers = sample(1:2, 1L), outputs = sample(list("output", all_outputs(id)), 1L)[[1L]])
      },
      generators = gen_shape(3L))
  }
})

test_that("a recurrent layer trains inside a learner", {
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    nn("lstm", hidden_size = 4) %>>%
    nn("squeeze", dim = 2) %>>%
    nn("head") %>>%
    po("torch_loss", t_loss("cross_entropy")) %>>%
    po("torch_optimizer", t_opt("adam")) %>>%
    po("torch_model_classif", epochs = 1, batch_size = 50)
  lrn = as_learner(graph)
  lrn$train(tsk("iris"))
  expect_prediction(lrn$predict(tsk("iris")))
})

test_that("only the requested final state can be an output channel", {
  task = tsk("iris")
  md = seq_graph(po("nn_lstm", hidden_size = 5, outputs = "c_n"))$train(task)
  expect_length(md, 1L)
  expect_equal(md[[1L]]$pointer_shape, c(NA, 1L, 5L))
  net = model_descriptor_to_module(md[[1L]])
  expect_equal(net(torch_randn(2, 4))$shape, c(2, 1, 5))

  # the final states are not computed when they are not requested
  md = seq_graph(po("nn_gru", hidden_size = 5))$train(task)
  expect_false(md[[1L]]$graph$pipeops$nn_gru$module$return_state)
})
