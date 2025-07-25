make_check_measures = function(task_type = NULL) {
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
    # CallbackSetHistory has a column 'epoch' in a data.table, where all other columns are the ids
    # of the measures
    if ("epoch" %in% ids(x)) {
      stopf("Measure must not have id 'epoch'.")
    }
    if (!is.null(task_type)) {
      # some measures have task_type NA, which means they work with all task types
      if (!all(map_lgl(map(x, "task_type"), function(x) task_type %in% x || (length(x) == 1L && is.na(x))))) {
        return(sprintf("Measures must support task type \"%s\".", task_type))
      }
    }
    f = function(x) "requires_learner" %in% x || "requires_model" %in% x
    if (any(map_lgl(map(x, "properties"), f))) {
      return("Measures must not require a learner or model.")
    }
    return(TRUE)
  }, task_type, .parent = topenv())

}

check_measures_regr = make_check_measures("regr")
check_measures_classif = make_check_measures("classif")
check_measures = make_check_measures()

epochs_aggr = function(x) as.integer(ceiling(mean(unlist(x))))

epochs_tune_fn = function(domain, param_vals) {
  checkmate::assert(
    checkmate::check_true(param_vals$patience > 0L),
    checkmate::check_true(domain$lower <= 1),
    combine = "and",
    .var.name = "When using early stopping, patience is set to a value > 0 and no lower bound for epochs is set."
  )
  domain$upper
}


paramset_torchlearner = function(task_type, jittable = FALSE) {
  check_measures = switch(task_type,
    regr = check_measures_regr,
    classif = check_measures_classif,
    make_check_measures(task_type)
  )

  param_set = ps(
    epochs                = p_int(tags = c("train", "validation", "internal_tuning", "required"), lower = 0L,
      aggr = epochs_aggr, in_tune_fn = epochs_tune_fn, disable_in_tune = list(patience = 0, measures_valid = list())),
    device                = p_fct(tags = c("train", "predict", "required"), levels = mlr_reflections$torch$devices, init = "auto"),
    num_threads           = p_int(lower = 1L, tags = c("train", "predict", "required", "threads"), init = 1L),
    num_interop_threads   = p_int(lower = 1L, tags = c("train", "predict", "required"), init = 1L),
    seed                  = p_int(tags = c("train", "predict", "required"), special_vals = list("random", NULL), init = "random"),
    # evaluation
    eval_freq             = p_int(lower = 1L, tags = c("train", "required"), init = 1L),
    measures_train        = p_uty(tags = c("train", "required"), custom_check = check_measures, init = list()),
    measures_valid        = p_uty(tags = c("train", "required"), custom_check = check_measures, init = list()),
    # early stopping
    patience              = p_int(lower = 0L, tags = c("train", "required"), init = 0L),
    min_delta             = p_dbl(lower = 0, tags = c("train", "required"), init = 0),
    # dataloader parameters
    batch_size            = p_int(tags = c("train", "predict", "required"), lower = 1L),
    shuffle               = p_lgl(tags = "train", default = FALSE, init = TRUE),
    sampler               = p_uty(tags = c("train", "predict")),
    batch_sampler         = p_uty(tags = c("train", "predict")),
    num_workers           = p_int(lower = 0, default = 0, tags = c("train", "predict")),
    collate_fn            = p_uty(tags = c("train", "predict"), default = NULL),
    pin_memory            = p_lgl(default = FALSE, tags = c("train", "predict")),
    drop_last             = p_lgl(tags = "train", default = FALSE),
    timeout               = p_dbl(default = -1, tags = c("train", "predict")),
    worker_init_fn        = p_uty(tags = c("train", "predict")),
    worker_globals        = p_uty(tags = c("train", "predict")),
    worker_packages       = p_uty(tags = c("train", "predict"), custom_check = check_character, special_vals = list(NULL)),
    tensor_dataset        = p_fct(levels = "device", init = FALSE, tags = c("train", "predict"), special_vals = list(FALSE, TRUE))
  )
  if (jittable) {
    param_set = c(
      param_set,
      ps(jit_trace = p_lgl(init = FALSE, tags = c("train", "required")))
    )
  }
  return(param_set)
}

