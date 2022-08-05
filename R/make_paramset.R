make_paramset = function(task_type, optimizer = NULL, loss = NULL) {
  param_set = ParamSetCollection$new(sets = list())

  ps1 = ps(
    batch_size            = p_int(tags = c("train", "predict", "required"), lower = 1L, default = 16L),
    epochs                = p_int(tags = c("train", "hotstart", "required"), lower = 0L),
    device                = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda", "meta"), default = "auto"), # nolint
    measures_train        = p_uty(tags = "train", custom_check = check_measures),
    measures_valid        = p_uty(tags = "train", custom_check = check_measures),
    augmentation          = p_uty(tags = "train"),
    callbacks             = p_uty(tags = "train", custom_check = check_callbacks),
    drop_last             = p_lgl(default = FALSE, tags = "train"),
    keep_last_prediction  = p_lgl(default = TRUE, tags = "train"),
    num_threads           = p_int(default = 1L, lower = 1L, tags = c("train", "predict", "threads")),
    shuffle               = p_lgl(default = TRUE, tags = "train"),
    early_stopping_rounds = p_int(default = 0L, tags = "train")
  )

  param_set$add(ps1)
  param_set$values = list(
    num_threads = 1L,
    drop_last = FALSE,
    shuffle = TRUE
  )

  if (!is.null(optimizer)) {
    optim_paramset = paramsets_optim$get(optimizer)
    optim_paramset$set_id = "opt"
    param_set$add(optim_paramset)
  }

  if (!is.null(loss)) {
    loss_paramset = paramsets_loss$get(loss)
    loss_paramset$set_id = "loss"
    param_set$add(loss_paramset)
  }
  return(param_set)
}

check_callbacks = function(x) {
  assert_true(all(map_lgl(x, function(x) inherits(x, "CallbackTorch"))))
}

check_network = function(x) {
  test_r6(x, "nn_Module")
}

check_measures = function(x) {
  if (!is.list(x)) {
    x = list(x)
  }
  if (test_list(x, types = "Measure")) {
    ids = map_chr(x, "id")
    if (test_names(ids, type = "unique")) {
      return(TRUE)
    }
  }

  "Parameter must be a Measure or list of Measures with valid ids."
}


check_network = function(x) {
  if (inherits(x, "nn_Module")) {
    "The network must be initialized by calling the function (and not with '$new()')."
  } else if (!test_class(x, "nn_module")) {
    "Must be a 'nn_module()'."
  } else {
    TRUE
  }
}

make_paramset_module = function(module) {
  pars = formalArgs(module)
  assert_true("..." %nin% pars)

  pars = lapply(
    pars,
    function(par) {
      if (par == "task") {
        NULL
      }
      ParamUty$new(id = paste0("net.", par), tags = "network")
    }
  )
  param_set = ParamSet$new(pars)
}

