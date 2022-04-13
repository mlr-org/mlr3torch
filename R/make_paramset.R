make_standard_paramset = function(task_type, optimizer, architecture = FALSE) {
  param_set = ps(
    optim_args = p_uty(tags = "train"),
    criterion = p_fct(levels = torch_reflections$loss[[task_type]], tags = "train"),
    crit_args = p_uty(tags = "train"),
    epochs = p_int(tags = c("train", "hotstart", "required"), lower = 0L),
    device = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda"), default = "auto"),
    batch_size = p_int(tags = c("train", "predict", "required"), lower = 1L, default = 16L),
    keep_last_prediction = p_lgl(default = TRUE, tags = "train"),
    early_stopping_set = p_fct(default = "test", levels = c("test", "train"), special_vals = list(NULL)),
    shuffle = p_lgl(default = TRUE, tags = "train"),
    drop_last = p_lgl(default = FALSE, tags = "train"),
    valid_split = p_dbl(default = 0.33, lower = 0, upper = 1, tags = "train")
  )
  param_set$values$valid_split = 0.33
  optim_paramset = make_paramset_optim(optimizer)
  param_set$add(optim_paramset)
  if (architecture) {
    param_set$add(ParamUty$new("architecture", tags = "train", custom_check = check_architecture))
  }
  return(param_set)
}


check_architecture = function(x) {
  if (test_r6(x, "Architecture") || test_r6(x, "nn_Module")) {
    return(TRUE)
  }
  return("Parameter 'architecture' must either be an mlr3torch::Architecture or torch::nn_Module.")
}
