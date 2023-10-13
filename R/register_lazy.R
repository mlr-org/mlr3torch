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

  Class = R6Class(classname,
    inherit = PipeOpTorchLazyTrafo,
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
  classname = paste0("PipeOpTaskPreprocLazy", capitalize(name))
  idname = paste0("preproc_", name)

  init_fun = crate(function(id = idname, param_vals = list()) {
    super$initialize(
      id = id,
      packages = packages,
      param_set = param_set,
      param_vals = param_vals,
      fn = fn
    )
  }, param_set, fn, idname, packages, .parent = topenv())

  Class = R6Class(classname,
    inherit = PipeOpTaskPreprocLazy,
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

#' @include PipeOpTorchLazyTrafo.R PipeOpTorchLazyPreproc.R
register_lazy = function(name, fn, param_set, shapes_out, packages, envir = parent.frame()) {
  register_trafo(name, fn, param_set, shapes_out, packages, envir = envir)
  register_preproc(name, fn, param_set, shapes_out, packages, envir = envir)

  return(NULL)
}

#' @title Lazy Preprocessing and Transformations
#' @name preproc_and_lazy
#' @section Available PipeOps:
#' The following
NULL


#' @name PipeOpTaskPreprocLazyResize
#' @alias PipeOpTorchLazyTrafoResize
#' @rdname preproc_and_lazy
register_lazy("resize", torchvision::transform_resize,
  packages = "torchvision",
  param_set = ps(
    size = p_uty(tags = c("train", "required")),
    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
      tags = "train", default = 2L
    )
  ),
  shapes_out = function(shapes_in, param_vals, task) {
    size = param_vals$size
    shape = shapes_in[[1L]]
    assert_true(length(shape) > 2)
    height = shape[[length(shape) - 1L]]
    width = shape[[length(shape)]]
    s = torchvision::transform_resize(torch_ones(c(1, height, width), device = "meta"), size = size)$shape[2:3]
    list(c(shape[seq_len(length(shape) - 2L)], s))
  }
)

#' @rawNamespace exportPattern("^PipeOpTaskPreprocLazy")
#' @rawNamespace exportPattern("^PipeOpTorchLazyTrafo")
