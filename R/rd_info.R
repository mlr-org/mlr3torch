rd_info_learner_torch = function(name, task_types = "classif, regr") {
  task_types = gsub(" ", "", task_types)
  task_types = strsplit(task_types, split = ",")[[1L]]
  predict_types = c()
  if ("classif" %in% task_types) {
    learner = lrn_classif = lrn(paste0("classif.", name))
    predict_types = c(predict_types,
      sprintf("  * classif: %s", paste0("'", lrn_classif$predict_types, "'", collapse = ", "))
    )
  }
  if ("regr" %in% task_types) {
    learner = lrn_regr = lrn(paste0("regr.", name))
    predict_types = c(predict_types,
      sprintf("  * regr: %s", paste0("'", lrn_regr$predict_types, "'", collapse = ", "))
    )
  }
  x = c("",
    sprintf("* Supported task types: %s", paste0("'", task_types, "'", collapse = ", ")),
    sprintf("* Predict Types:"),
    predict_types,
    sprintf("* Feature Types: %s", rd_format_string(learner$feature_types)),
    sprintf("* Required Packages: %s", rd_format_packages(learner$packages))
  )
  paste(x, collapse = "\n")
}

rd_info_task_torch = function(name, missings) {
  obj = tsk(name)
  x = c("",
    sprintf("* Task type: %s", rd_format_string(obj$task_type)),
    sprintf("* Properties: %s", rd_format_string(obj$properties)),
    sprintf("* Has Missings: %s", if (missings) "yes" else "no"),
    sprintf("* Target: %s", rd_format_string(obj$target_names)),
    sprintf("* Features: %s", rd_format_string(obj$feature_names)),
    sprintf("* Data Dimension: %ix%i", obj$backend$nrow, obj$backend$ncol)
  )
  paste(x, collapse = "\n")
}
