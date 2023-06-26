make_check_measures = function(task_type) {
  env = parent.env(environment())
  crate(function(x) {
    if (is.null(x)) {
      return(TRUE)
    }
    if (!is.list(x)) {
      x = list(x)
    }
    msg = check_list(x, types = "Measure")
    if (!isTRUE(msg)) return(msg)

    if (!test_names(ids(x), type = "unique")) {
      return("IDs of measures must be unique.")
    }
    # some measures have task_type NA, which means they work with all task types
    if (!all(map_lgl(map(x, "task_type"), function(x) task_type %in% x || (length(x) == 1L && is.na(x))))) {
      return(sprintf("Measures must support task type \"%s\".", task_type))
    }
    f = function(x) "requires_learner" %in% x || "requires_model" %in% x
    if (any(map_lgl(map(x, "properties"), f))) {
      return("Measures must not require a learner or model.")
    }
    return(TRUE)
  }, task_type = task_type, .parent = env)

}

check_measures_regr = make_check_measures("regr")
check_measures_classif = make_check_measures("classif")

paramset_torchlearner = function(task_type) {
  check_measures = switch(task_type,
    regr = check_measures_regr,
    classif = check_measures_classif,
    stopf("Unsupported task type \"%s\".", task_type)
  )

  param_set = ps(
    batch_size            = p_int(tags = c("train", "predict", "required"), lower = 1L),
    epochs                = p_int(tags = c("train", "required"), lower = 0L),
    device                = p_fct(tags = c("train", "predict", "required"), levels = mlr_reflections$torch$devices),
    measures_train        = p_uty(tags = c("train", "required"), custom_check = check_measures),
    measures_valid        = p_uty(tags = c("train", "required"), custom_check = check_measures),
    drop_last             = p_lgl(tags = c("train", "required")),
    shuffle               = p_lgl(tags = c("train", "required")),
    num_threads           = p_int(lower = 1L, tags = c("train", "predict", "required", "threads")),
    seed                  = p_int(tags = c("train", "predict", "required"), special_vals = list("random"))
  )
  param_set$values = list(
    device         = "auto",
    measures_train = list(),
    measures_valid = list(),
    num_threads    = 1L,
    drop_last      = FALSE,
    shuffle        = TRUE,
    seed           = "random"
  )
  return(param_set)
}
