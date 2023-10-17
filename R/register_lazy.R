register_trafo = function(name, fn, param_set, shapes_out = NULL, packages, envir) {
  classname = paste0("PipeOpTorchTrafo", capitalize(name))
  idname = paste0("trafo_", name)

  init_fun = crate(function(id = idname, param_vals = list()) {
    super$initialize(
      id = id,
      packages = packages,
      param_set = param_set,
      param_vals = param_vals,
      fn = fn
    )
  }, param_set, fn, idname, packages, .parent = topenv())
  body(init_fun)[[2]][[4]] = param_set
  attributes(init_fun) = NULL

  Class = R6Class(classname,
    inherit = PipeOpTorchTrafo,
    public = list(
      initialize = init_fun
    ),
    private = list(
      .shapes_out = shapes_out
    )
  )

  assign(classname, Class, envir = envir)
  register_po(idname, Class)

  return(NULL)
}

register_preproc = function(name, fn, param_set, shapes_out, packages, envir) {
  classname = paste0("PipeOpPreprocTorch", capitalize(name))
  idname = paste0("preproc_", name)

  init_fun = crate(function(id = idname, param_vals = list()) {
    super$initialize(
      id = id,
      packages = packages,
      param_set = ps,
      param_vals = param_vals,
      fn = fn
    )
  }, fn, packages, .parent = topenv())
  # param_set is already an expression
  body(init_fun)[[2]][[4]] = param_set
  attributes(init_fun) = NULL

  Class = R6Class(classname,
    inherit = PipeOpTaskPreprocTorch,
    public = list(
      initialize = init_fun
    ),
    private = list(
      .shapes_out = crate(function(...))
    )
  )

  assign(classname, Class, envir = envir)
  register_po(idname, Class)

  return(NULL)

}

#' @include PipeOpTaskPreprocTorch.R PipeOpTorchTrafo.R
register_lazy = function(name, fn, param_set, shapes_out, packages, envir = parent.frame()) {
  register_trafo(name, fn, substitute(param_set), shapes_out, packages, envir = envir)
  register_preproc(name, fn, substitute(param_set), shapes_out, packages, envir = envir)

  return(NULL)
}
