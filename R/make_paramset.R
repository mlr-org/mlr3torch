make_standard_paramset = function(task_type) {
  ps(
    architecture = p_uty(tags = "train", custom_check = check_architecture),
    optimizer = p_fct(levels = torch_reflections$optimizer, tags = "train"),
    optimizer_args = p_uty(tags = "train"),
    criterion = p_fct(levels = torch_reflections$loss[[task_type]], tags = "train"),
    criterion_args = p_uty(tags = "train"),
    epochs = p_int(tags = c("train", "hotstart", "required"), lower = 0L),
    device = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda"), default = "auto"),
    batch_size = p_int(tags = c("train", "predict", "required"), lower = 1L, default = 16L),
    keep_last_prediction = p_lgl(default = TRUE, tags = "train"),
    early_stopping_set = p_fct(default = "test", levels = c("test", "train"), special_vals = list(NULL)),
    shuffle = p_lgl(default = TRUE, tags = "train"),
    drop_last = p_lgl(default = FALSE, tags = "train")
    # otherwise duplicate work
    # in resample() (we have the validation anyway)
  )
}

make_paramset_criterion = function(task_type) {
  # TODO: Finish this
  # param = switch(task_type,
  #   classif = p_fct(default = "bce", levels = torch_reflections$loss$classif),
  #   regr = p_fct(default = "mse", levels = torch_reflections$loss$regr),
  #   stopf("Task type %s not (yet) supported.", task_type)
  # )
  # ps(criterion = param)
  ps()
}


check_architecture = function(x) {
  if (test_r6(x, "Architecture") || test_r6(x, "nn_Module")) {
    return(TRUE)
  }
  return("Parameter 'architecture' must either be an mlr3torch::Architecture or torch::nn_Module.")
}
