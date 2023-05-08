paramset_torchlearner = function() {
  ps(
    batch_size            = p_int(tags = c("train", "predict", "required"), lower = 1L),
    epochs                = p_int(tags = c("train", "required"), lower = 0L),
    device                = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda"), default = "auto"),
    measures_train        = p_uty(tags = "train", custom_check = check_measures),
    measures_valid        = p_uty(tags = "train", custom_check = check_measures),
    # augmentation          = p_uty(tags = "train"),
    drop_last             = p_lgl(default = FALSE, tags = "train"),
    num_threads           = p_int(default = 1L, lower = 1L, tags = c("train", "predict", "threads")),
    shuffle               = p_lgl(default = TRUE, tags = "train"),
    early_stopping_rounds = p_int(default = 0L, tags = "train"),
    seed                  = p_int(tags = c("train", "predict"))
  )
}

check_callbacks = function(x) {
  if (!is.list(x)) {
    x = list(x)
  }
  ids = map_chr(x, "id")
  if (anyDuplicated(ids)) {
    return("The ids of callbacks must be unique.")
  }
  msg = check_list(x, types = "TorchCallback", any.missing = FALSE)
  if (is.character(msg)) {
    return(msg)
  }

  if (any(map_lgl(x, function(cb) !is.null(cb$state)))) {
    return(sprintf("Callbacks must not be trained."))
  }

  return(TRUE)
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
