dl_paramset = function(network = TRUE, architecture = TRUE) {
  param_set = ps(
    optimizer = p_uty(tags = "train"),
    optimizer_args = p_uty(tags = "train"),
    criterion = p_uty(tags = "train"),
    criterion_args = p_uty(tags = "train"),
    n_epochs = p_int(tags = "train", lower = 0L),
    device = p_fct(tags = c("train", "predict"), levels = c("cpu", "cuda"), default = "cpu"),
    batch_size = p_int(tags = c("train", "predict"), lower = 1L, default = 16L),
    architecture = p_uty(tags = c("train", "required"))
  )

  # if (architecture) {
  #   # TODO: Make architecture and network exclusive
  #   param_set$add(
  #     "architecture", p_uty(tags = "train", custom_check = check_network)
  #   )
  # }
  #
  # if (network) {
  #   param_set$add(
  #     "network", p_uty(tags = "train", custom_check = check_network)
  #   )
  # }
  return(param_set)
}


optimizer_check = function(x) {
  valid = test_character(par, any.missing = FALSE, len = 1, min.len = 1) ||
    test_r6(par, classes = "torch_Optimizer")
  if (test_character(par, any.missing = FALSE, len = 1, min.len = 1)) {
    valid = sprintf("optim_%s", par) %in% getNamespaceExports("torch")
    if (valid) {
      return(TRUE)
    } else {
      return(sprintf("Optimizer optim_%s not exported from package torch.", param))
    }
  }
  if (test_r6(par, "torch_Optimizer")) {
    return(TRUE)
  }
  return("Parameter must be a character specifying torch_* or torch_Optimizer.")
}

optimizer_trafo = function(x) {
  if (is.character(par)) {
    optimizer = mlr3misc::invoke(get(sprintf("optim_%s", par),
      envir = getNamespace("torch")))
    return(optimizer)
  }
  return(par)
}

criterion_check = function(x) {
  valid = test_choice(par, choices = c("mse", "cross_entropy"))
  if (valid) {
    return(TRUE)
  }
  return("Invalid criterion.")
}

criterion_trafo = function(x) {
  mlr3misc::invoke(get(sprintf("nn_%s_loss", par),
    envir = getNamespace("torch")))
}

network_check = function(x) {
  stop("Not implemented yet.")
  # if (test_r6(x, "NeuralNetwork")) {
  #   return(TRUE)
  # }
  # return("Parameter network must be an instance of class NeuralNetwork.")
}
architecture_check = function(x) {
  if (test_r6(x, "Architecture")) {
    return(TRUE)
  }
  return("Parameter architecture must be an instance of class Architecture")
}
