roxy_param_id = function(id = NULL) {
  paste0(
    "* `id` :: `character(1)`\\cr The id for the object.",
    if (!is.null(id)) sprintf(" The default is \"%s\".", id)
  )
}

roxy_pipeop_torch_construction = function(class, id = NULL) {
  sprintf("\\code{PipeOpTorch%s$new(id%s, param_vals = list())}", class, if (is.null(id)) "" else sprintf("= \"%s\"", id)) # nolint
}

roxy_pipeop_torch_construction = function(class, id = NULL) {
  sprintf("\\code{PipeOpTorch%s$new(id%s, param_vals = list())}", class, if (is.null(id)) "" else sprintf("= \"%s\"", id)) # nolint
}


roxy_pipeop_torch_param_id = function(id = NULL) {
  sprintf(
    "* \\code{id} :: \\code{character(1)}\\cr Identifier of the resulting object.%s",
    if (is.null(id))  "" else sprintf("Default is \"%s\"", id)
  )
}

roxy_pipeop_torch_fields_default = function() { # nolint
  "Only fields inherited from [\\code{PipeOpTorch}]/[\\code{PipeOp}]."
}

roxy_pipeop_torch_methods_default = function() { # nolint
  "Only methods inherited from [\\code{PipeOpTorch}]/[\\code{PipeOp}]."
}

roxy_param_param_vals = function() {
  "* \\code{param_vals} :: named \\code{list()}\\cr List of hyperparameter settings to overwrite the initial values. Default is  \\code{list()}." # nolint
}

roxy_param_module_generator = function() {
  "* \\code{module_generator} :: \\code{nn_module_generator}\\cr The torch module generator."
}

roxy_param_param_set = function() {
  "* \\code{param_set} :: \\code{paradox::ParamSet}\\cr The parameter set."
}

roxy_pipeop_torch_license = function() {
  paste(
    "Parts of this documentation have been copied or adapted from the R package [torch], that comes under the",
    " MIT license, which is included in the help page of [\\code{mlr3torch}], as well as in the top-level folder of",
    " the package source.", sep = "\n"
  )
}

roxy_pipeop_torch_channels_default = function() { # nolint
  paste0(
    "One input channel called \\code{\"input\"} and one output channel called",
    "‘\"output\"’. For an explanation see [\\code{PipeOpTorch}]."
  )
}

roxy_pipeop_torch_state_default = function() { # nolint
 "The state is the value calculated by the public method \\code{shapes_out()}."
}


roxy_pipeop_torch_format = function() {
  "[\\code{R6Class}] inheriting from [\\code{PipeOpTorch}]/[\\code{PipeOp}]."
  # "Blablabla"
}


roxy_param_packages = function() {
  "* \\code{packages}` :: named \\code{list()}\\cr List of packages settings. Default is  \\code{list()}."
}

roxy_param_innum = function() {

}

roxy_construction = function(x) {
  find_obj_with_init = function(x) {
    if (is.null(x)) {
      # This is the case where no initialize method is found
      return(NULL)
    } else if (!is.null(x$public_methods$initialize)) {
      return(x)
    } else {
      find_obj_with_init(x$get_inherit())
    }
  }
  obj_with_init = find_obj_with_init(x)
  if (is.null(obj_with_init)) {
    # None of the parents or the object itself has an initialize method
    obj_with_init = function() NULL
  } else {
    init = obj_with_init$public_methods$initialize

  }
  fs = formals(init)
  args = names(fs)
  defaults = unname(fs)
  defaults = map_chr(defaults, function(x) deparse(x))

  inner = paste0(paste0(args, ifelse(defaults == "", "", " = "), defaults), collapse = ", ")

  sprintf("\\code{%s$new(%s)}", x$classname, inner)
}
