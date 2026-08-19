# Fixtures for the generic torch task, shared by test_TaskTorch.R and test_PredictionTorch.R.
#
# a small MLP whose output layer is sized through `output_dim_for()`
tt_module = nn_module("tt_module",
  initialize = function(task) {
    self$net = nn_sequential(
      nn_linear(length(task$feature_names), 10L), nn_relu(),
      nn_linear(10L, output_dim_for(task))
    )
  },
  forward = function(x) self$net(x)
)

# A `TaskTorch` specifies none of this itself, so the tests state the encodings that the built-in
# task types use. `tt_task()` and `tt_learner()` fill them in, the way a user would, which keeps the
# tests below about the plumbing rather than about the encodings.
# The storage of the first target column, read from the task's metadata rather than from a row: a
# task filtered to zero rows has no row to read, and both functions below are called on one when
# `mlr3` builds an empty prediction (see `create_empty_prediction_data()`).
tt_target_info = function(task) {
  task$col_info[list(task$target_names[1L]), on = "id"]
}

tt_odim = function(task) {
  target = task$target_names
  info = tt_target_info(task)
  if (length(target) == 1L && info$type == "factor") length(info$levels[[1L]]) else length(target)
}

tt_enc = function(task, predict_tensor, predict_type) {
  target = task$target_names
  info = tt_target_info(task)
  if (length(target) == 1L && info$type == "factor") {
    levs = info$levels[[1L]]
    response = factor(as.integer(with_no_grad(predict_tensor$argmax(dim = 2L))$to(device = "cpu")),
      levels = seq_along(levs), labels = levs)
    prob = if (predict_type == "prob") {
      p = as.matrix(with_no_grad(nnf_softmax(predict_tensor, dim = 2L))$to(device = "cpu"))
      colnames(p) = levs
      p
    }
    return(list(response = response, prob = prob))
  }
  if (info$type == "logical") {
    prob = as.matrix(with_no_grad(nnf_sigmoid(predict_tensor))$to(device = "cpu"))
    colnames(prob) = target
    response = prob > 0.5
    if (length(target) == 1L) {
      response = as.logical(response)
      prob = as.numeric(prob)
    }
    return(list(response = response, prob = if (predict_type == "prob") prob))
  }
  predict_tensor = with_no_grad(predict_tensor)$to(device = "cpu")
  if (length(target) == 1L) {
    return(list(response = as.numeric(predict_tensor)))
  }
  response = as.matrix(predict_tensor)
  colnames(response) = target
  list(response = response)
}

tt_bg = function(data) {
  if (ncol(data) == 1L && is.factor(data[[1L]])) {
    return(torch_tensor(as.integer(data[[1L]]), dtype = torch_long()))
  }
  torch_tensor(1 * as.matrix(data), dtype = torch_float())
}

tt_task = function(x, target = character(0), id = "t", ...) {
  args = list(...)
  if (is.null(args$output_dim)) args$output_dim = tt_odim
  if (is.null(args$prediction_encoder)) args$prediction_encoder = tt_enc
  invoke(as_task_torch, x = x, target = target, id = id, .args = args)
}

tt_learner = function(loss, ...) {
  args = insert_named(list(epochs = 3L, batch_size = 16L, target_batchgetter = tt_bg), list(...))
  invoke(lrn, "torch.module",
    module_generator = tt_module,
    ingress_tokens = list(x = ingress_num()),
    loss = loss,
    .args = args
  )
}

tt_data = function(n = 40L) {
  withr::with_seed(1L, data.table(
    x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n)
  ))
}
