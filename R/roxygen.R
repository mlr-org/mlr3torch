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
  "* \\code{param_vals}` :: named \\code{list()}\\cr List of hyperparameter settings. Default is  \\code{list()}."
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

roxy_consturction = function(obj) {
  sprintf("%s$new()")


}

roxy_param_packages = function() {

}

roxy_param_innum = function() {

}
