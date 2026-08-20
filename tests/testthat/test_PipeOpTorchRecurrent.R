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

test_that("'return_state' determines the output channels", {
  expect_equal(po("nn_lstm", hidden_size = 5)$output$name, "output")
  expect_equal(po("nn_lstm", hidden_size = 5, return_state = TRUE)$output$name,
    c("output", "h_n", "c_n"))
  # only an LSTM carries a cell state
  expect_equal(po("nn_gru", hidden_size = 5, return_state = TRUE)$output$name, c("output", "h_n"))
  expect_equal(po("nn_rnn", hidden_size = 5, return_state = TRUE)$output$name, c("output", "h_n"))

  # it is a construction argument and not a hyperparameter
  expect_true("return_state" %nin% po("nn_lstm")$param_set$ids())
  expect_false(po("nn_lstm", hidden_size = 5)$phash ==
    po("nn_lstm", hidden_size = 5, return_state = TRUE)$phash)
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
    expect_shape_inference(id, list(hidden_size = 5, return_state = TRUE), c(2, 7, 4))
    expect_shape_inference(id,
      list(hidden_size = 5, return_state = TRUE, num_layers = 2, bidirectional = TRUE), c(2, 7, 4))
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
          num_layers = sample(1:2, 1L), return_state = sample(c(TRUE, FALSE), 1L))
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
