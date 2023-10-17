#' @include register_lazy.R
NULL

# Where I have left off:
# I split the class for the lazy tensor preprocessing / transformation into two clases
# I also decided to create the corresponding classes programmatically, otherwise we just have to write so much code
# there will be one or two (either one for both preproc and trafo or seperately) manual pages that document how to
# use the pipeops + a pointer to the parent class + a list with available pipeops
# This refactoring now needs to be done + the resulting pipeops than tested

#' @rawNamespace exportPattern("^PipeOpTaskPreprocTorch")
#' @rawNamespace exportPattern("^PipeOpTorchTrafo")
NULL

#' @title Lazy Preprocessing and Transformations
#' @name preproc_and_lazy
#' @section Available PipeOps:
#' The following
NULL

#' @name PipeOpTaskPreprocTorchResize
#' @alias PipeOpTorchTrafoResize
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
