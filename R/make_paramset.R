make_paramset = function(task_type, optimizer, loss, architecture = FALSE) {
  param_set = ParamSetCollection$new(sets = list())
  ps1 = ps(
    augmentation = p_uty(tags = "train"),
    batch_size = p_int(tags = c("train", "predict", "required"), lower = 1L, default = 16L),
    callbacks = p_uty(tags = "train", custom_check = check_callbacks),
    device = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda", "meta"),
      default = "auto"),
    drop_last = p_lgl(default = FALSE, tags = "train"),
    epochs = p_int(tags = c("train", "hotstart", "required"), lower = 0L),
    keep_last_prediction = p_lgl(default = TRUE, tags = "train"),
    measures = p_uty(tags = "train"),
    num_threads = p_int(default = 1L, lower = 1L, tags = c("train", "predict", "threads")),
    shuffle = p_lgl(default = TRUE, tags = "train"),
    valid_split = p_dbl(default = 0.33, lower = 0, upper = 1, tags = "train")
  )
  if (architecture) {
    ps1$add(ParamUty$new("architecture", tags = "train", custom_check = check_architecture))
  }
  param_set$add(ps1)
  param_set$values = list(
    valid_split = 0.33,
    num_threads = 1L,
    drop_last = FALSE,
    shuffle = TRUE,
    measures = list()
  )

  optim_paramset = paramsets_optim$get(optimizer)
  optim_paramset$set_id = "opt"
  loss_paramset = paramsets_loss$get(loss)
  loss_paramset$set_id = "loss"

  param_set$add(optim_paramset)
  param_set$add(loss_paramset)
  return(param_set)
}

check_callbacks = function(x) {
  assert_true(all(map_lgl(x, function(x) inherits(x, "CallbackTorch"))))
}

check_architecture = function(x) {
  if (test_r6(x, "Architecture") || test_r6(x, "nn_Module")) {
    return(TRUE)
  }
  return("Parameter 'architecture' must either be an mlr3torch::Architecture or torch::nn_Module.")
}

check_measures = function(x) {
  assert_list(x)
  nms = names(x)
  if (!is.null(nms)) {
    assert_subset(nms, c("", "train", "valid"))
  }
}
